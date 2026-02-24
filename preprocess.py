import os
import torch
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle, Polygon
from matplotlib.backends.backend_pdf import PdfPages
from transformers import pipeline, AutoModelForMaskGeneration, AutoProcessor
import colorsys

class VectorAnnotator:
    def __init__(self, 
                 detector_id="IDEA-Research/grounding-dino-base", 
                 segmenter_id="facebook/sam-vit-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading models on {self.device}...")
        
        try:
            self.detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=self.device)
            self.sam_model = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(self.device)
            self.sam_processor = AutoProcessor.from_pretrained(segmenter_id)
            print("Models loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def process_image(self, image_path, output_path, text_prompt, box_threshold=0.3):
        """处理单张图片并生成PDF"""
        try:
            image = Image.open(image_path).convert("RGB")
        except IOError:
            print(f"Cannot open image: {image_path}")
            return

        # 1. 检测 (Detection)
        labels = [text_prompt if text_prompt.endswith(".") else f"{text_prompt}."]
        detections = self.detector(image, candidate_labels=labels, threshold=box_threshold)
        
        if not detections:
            print(f"No objects found in {os.path.basename(image_path)}")
            return

        # 2. 分割 (Segmentation)
        input_boxes = [[d['box']['xmin'], d['box']['ymin'], d['box']['xmax'], d['box']['ymax']] for d in detections]
        inputs = self.sam_processor(images=image, input_boxes=[input_boxes], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        
        masks = self.sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0][:, 0, :, :].cpu().numpy()

        # 3. 生成PDF (Vector PDF Generation)
        self._create_pdf(image, detections, masks, output_path)
        print(f"Saved: {output_path}")

    def _create_pdf(self, image, detections, masks, output_path):
        """生成多页矢量PDF"""
        height, width = image.height, image.width
        dpi = 100
        colors = self._get_colors(len(detections))

        with PdfPages(output_path) as pdf:
            # Page 1: 仅边界框
            self._save_plot_page(pdf, image, detections, None, colors, width, height, dpi)
            
            # Page 2: 掩码 + 边界框
            self._save_plot_page(pdf, image, detections, masks, colors, width, height, dpi)

            # Page 3+: 单个物体裁剪
            img_np = np.array(image)
            for i, (det, mask) in enumerate(zip(detections, masks)):
                self._save_crop_page(pdf, img_np, det['box'], mask, width, height, dpi)

    def _save_plot_page(self, pdf, image, detections, masks, colors, w, h, dpi):
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
        ax.imshow(image)
        ax.axis('off')

        for i, det in enumerate(detections):
            # 绘制掩码 (Vector)
            if masks is not None:
                mask_uint8 = masks[i].astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    poly = Polygon(contour.reshape(-1, 2), facecolor=colors[i], edgecolor='none', alpha=0.5)
                    ax.add_patch(poly)

            # 绘制边界框 (Vector)
            b = det['box']
            rect = Rectangle((b['xmin'], b['ymin']), b['xmax']-b['xmin'], b['ymax']-b['ymin'],
                             linewidth=1.5, edgecolor=colors[i], facecolor='none')
            ax.add_patch(rect)

        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _save_crop_page(self, pdf, img_np, box, mask, w, h, dpi):
        # 创建透明背景图
        rgba = np.dstack((img_np, np.zeros(mask.shape, dtype=np.uint8)))
        rgba[mask, 3] = 255
        
        # 转白底并裁剪
        obj_img = Image.fromarray(rgba).convert("RGB") # 简单转RGB会变黑底，需处理透明度
        bg = Image.new("RGB", obj_img.size, (255, 255, 255))
        bg.paste(Image.fromarray(rgba), mask=Image.fromarray(rgba).split()[3])
        
        crop = bg.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
        
        fig, ax = plt.subplots(figsize=(crop.width/dpi, crop.height/dpi), dpi=dpi)
        ax.imshow(crop)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    @staticmethod
    def _get_colors(n):
        colors = [colorsys.hsv_to_rgb(i/n, 0.9, 0.9) for i in range(n)]
        random.shuffle(colors)
        return colors

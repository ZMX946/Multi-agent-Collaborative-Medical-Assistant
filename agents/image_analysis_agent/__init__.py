from .image_classifier import ImageClassifier
from .chest_xray_agent.covid_chest_xray_inference import ChestXRayClassification
from .brain_tumor_agent.brain_tumor_inference import BrainTumor3DUNet
from .skin_lesion_agent.skin_lesion_inference import SkinLesionSegmentation

class ImageAnalysisAgent:
    """
    负责处理图像上传、将其分类为医疗或非医疗用途，并确定其类型的代理程序。
    """
    
    def __init__(self, config):
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)
        self.chest_xray_agent = ChestXRayClassification(model_path=config.medical_cv.chest_xray_model_path)
        self.brain_tumor_agent = BrainTumor3DUNet(model_path=config.medical_cv.brain_tumor_model_path)
        self.skin_lesion_agent = SkinLesionSegmentation(model_path=config.medical_cv.skin_lesion_model_path)
        self.skin_lesion_segmentation_output_path = config.medical_cv.skin_lesion_segmentation_output_path
    
    # 分类图像
    def analyze_image(self, image_path: str) -> str:
        """将图像分类为医疗或非医疗图像，并确定其类型。"""
        return self.image_classifier.classify_image(image_path)
    
    # 胸部X光造影剂
    def classify_chest_xray(self, image_path: str) -> str:
        return self.chest_xray_agent.predict(image_path)
    
    # brain tumor agent
    def classify_brain_tumor(self, image_path: str) -> str:
        return self.brain_tumor_agent.predict(image_path)
    
    # 皮肤病变剂
    def segment_skin_lesion(self, image_path: str) -> str:
        return self.skin_lesion_agent.predict(image_path, self.skin_lesion_segmentation_output_path)

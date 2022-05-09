import cv2
import numpy as np
import streamlit as st
from PIL import Image

import torch,torchvision
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# -----------------ФУНКЦИИ ДЛЯ ПРИЛОЖЕНИЯ-----------------
@st.cache(persist=True)
def initialization():
    """Загружаем модель и config для сегментации.
    
    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        classes_names (list: str): Classes available for the model of interest.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.
        
    """
    cfg = get_cfg()
    # Устанавливаем cpu, как device для модели.
    # Если установлена CUDA удалить эту строку:
    cfg.MODEL.DEVICE = 'cpu'
    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    # загружаем модель:
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # Устанавливаем порог определения объектов:
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  
    # Подгружаем веса в модель, см.больше тут: https://dl.fbaipublicfiles...
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # Подгружаем названия классов для определения:
    classes_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    # Устанавливаем predictor:
    predictor = DefaultPredictor(cfg)

    return cfg, classes_names, predictor

def inference(predictor, img):
    return predictor(img)

@st.cache
def output_image(cfg, img, outputs):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image() #[:, :, ::-1]

    return processed_img

#@st.cache
def retrieve_image():
    uploaded_img = st.file_uploader("Загрузите изображение ниже", type=['jpg', 'jpeg', 'png'])
    uploaded_img_cache = None
    if uploaded_img is not None and uploaded_img != uploaded_img_cache:
        uploaded_img_cache = uploaded_img
        print(uploaded_img)
        return image(uploaded_img)
        # file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        # try:
        #     img = cv2.imdecode(file_bytes, 1)
        # except:
        #     pass
        # return img
    
@st.cache
def image(uploaded_img):
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    try:
        img = cv2.imdecode(file_bytes, 1)
        return img
    except:
        return None
    return img

@st.cache
def discriminate(outputs, classes_to_detect):
    """Select which classes to detect from an output.

    Get the dictionary associated with the outputs instances and modify
    it according to the given classes to restrict the detection to them

    Args:
        outputs (dict):
            instances (detectron2.structures.instances.Instances): Instance
                element which contains, among others, "pred_boxes", 
                "pred_classes", "scores" and "pred_masks".
        classes_to_detect (list: int): Identifiers of the dataset on which
            the model was trained.

    Returns:
        ouputs (dict): Same dict as before, but modified to match
            the detection classes.

    """
    pred_classes = np.array(outputs['instances'].pred_classes)
    # Элементы, соответствующие выбранным классам *classes_to_detect*
    mask = np.isin(pred_classes, classes_to_detect)
    # Получаем индексы
    idx = np.nonzero(mask)

    # Get Instance values as a dict and leave only the desired ones
    out_fields = outputs['instances'].get_fields()
    for field in out_fields:
        out_fields[field] = out_fields[field][idx]

    return outputs

# -----------------САМО ПРИЛОЖЕНИЕ-----------------
def main():
    # Инициализация
    cfg, classes, predictor = initialization()

    # Streamlit initialization
    st.markdown('''<h1 style='text-align: center; color: #9F2B68;'
            >Instance segmentation</h1>''', 
            unsafe_allow_html=True)
    st.write("""
        Данное приложение позволяет познакомиться с основами сегментации изображений. 
        \nДля этого используется модель *mask_rcnn_R_50_FPN_3x*, обученная распознавать и выделять на фото 80 классов изображений.
        \n**Используемые библиотеки:** [cv2](https://opencv.org/),[Detectron2](https://detectron2.readthedocs.io/en/latest/), [Streamlit](https://docs.streamlit.io/library/get-started).

        \n**Полезно почитать:** [О библиотеке Detectron2](https://github.com/facebookresearch/detectron2), [Семантическая и инстанс-сегменация](https://neurohive.io/ru/osnovy-data-science/semantic-segmention/).
        \nДанные подготовили сотрудники ЛИА РАНХиГС.
        """)

    img_pipeline = Image.open('Pipeline_for_instance_segmentation.png') #
    st.image(img_pipeline, use_column_width='auto', caption='Общий пайплайн для приложения') #width=450

    # ----------------О проекте----------------#

    expander_bar = st.expander("Что такое сегментация изображений?")
    expander_bar.markdown(
        """
        В широком смысле **«сегментация»**, — это разделение чего-либо на несколько частей, либо одна из таких частей. 
        \nВ нашем случае сегментация — это задача в компьютерном зрении, которая позволяет разделить цифровое изображение на разные части (сегменты) в соответствии с тем, какому объекту какие пиксели принадлежат. Таким образом мы получаем попиксельную маску объекта.

        \n**Некоторыми практическими применениями сегментации изображений являются:**
        \n* Исследование медицинских изображений
        \n* Выделение объектов на спутниковых снимках
        \n* Распознавание лиц
        \n* Распознавание отпечатков пальцев
        \n* Системы управления дорожным движением
        \n* Обнаружение стоп-сигналов
        \n* Машинное зрение
        \n* Распараллеливание информационных потоков при передаче изображений высокого разрешения

        \n**Виды сегментации**:

        \n1. **Семантическая сегментация (Semantic segmentation)** — определяет принадлежность наборов пикселей на изображении к определенным классам объектов (например, кошки, собаки, люди, цветы, автомобили и т.д.). 
        \n2. **Инстанс-сегментация (Instance segmentation)** — в отличие от семантической сегментации, в этой задаче каждый объект внутри одного класса выделяется отдельными сегментами. Например, если на изображении пять кошек, две собаки и десять растений, семантическая сегментация просто выделит все области, на которых есть кошки, собаки или растения, не разделяя отдельные объекты внутри каждого класса (определит, что на изображении есть кошки, собаки и растения), в то время как инстанс-сегментация выделит каждую кошку, собаку и растение как отдельный объект. 
        \n3. **Паноптическая сегментация (Panoptic segmentation)** — объединяет задачи семантической и инстанс-сегментации. Также в задаче паноптической сегментации каждому пикселю изображения должна быть присвоена ровно одна метка.
        
        """
    )
    # ----------Зона пользователя----------
    st.markdown('''<h2 style='text-align: center; color: black;'
            >Инстанс-сегментация изображений</h2>''', 
            unsafe_allow_html=True)
    st.markdown("""
        Вам необходимо выбрать классы для сегментации, а так же загрузить само изображение.
    \nЕсли не указывать определённый класс, модель автоматически распознает и выделит на фото все найденные классы из 80 имеющихся.
    """)
    exp_bar = st.expander("Для справки: классы объектов, которые распознаёт mask_rcnn_R_50_FPN_3x")
    exp_bar.code(classes)
    ## Выбор классов для моделей
    classes_to_detect = st.multiselect("Выберите классы для определения на изображении:", classes, ['person'])
    
    # mask = np.isin(classes, classes_to_detect)
    # class_idxs = np.nonzero(mask)

    # Место для изображения
    img_placeholder = st.empty()

    # ----------Обработка и получение результатата----------
    uploaded_img = st.file_uploader("Загрузите изображение ниже:", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        # try:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # ДЕТЕКЦИЯ МОДЕЛЬЮ:
        outputs = inference(predictor, img)
        if not classes_to_detect:
            out_image = output_image(cfg, img, outputs)
        else:
            mask = np.isin(classes, classes_to_detect)
            class_idxs = np.nonzero(mask)
            outputs = discriminate(outputs, class_idxs)
            out_image = output_image(cfg, img, outputs)
        st.image(out_image, caption='Загруженное и обработанное изображение', use_column_width=True)        
        # except:
        #     st.subheader("Пожалуйста, загрузите изображение заново, чтобы увидеть изменения")

        # не надо выбирать:
        # outputs = inference(predictor, img)
        # out_image = output_image(cfg, img, outputs)

        # НАДО выбирать:
        # outputs = inference(predictor, img)
        # outputs = discriminate(outputs, class_idxs)
        # out_image = output_image(cfg, img, outputs)

# -----------------ЗАПУСКАЕМ-----------------
if __name__ == '__main__':
    main()
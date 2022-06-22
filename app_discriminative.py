import cv2
import numpy as np
import streamlit as st
import time
# from datetime import datetime
# import asyncio
# from time import time, clock
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
# st.set_page_config(layout="wide")
def main():
    # ----------Инициализация----------
    cfg, classes, predictor = initialization()

    # ----------Обработка и получение результатата----------
    def image_segmentation(uploaded_img, classes_to_detect): # первый аргумент - uploaded_img
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # st.write(img.shape)
        # st.write(type(img))
        # ДЕТЕКЦИЯ МОДЕЛЬЮ:
        outputs = inference(predictor, img)
        if not classes_to_detect:
            out_image = output_image(cfg, img, outputs)
            num_of_instances = len(outputs['instances']) # сколько объектов найдено
            what_instances = outputs['instances'].pred_classes # какие классы найдены
        else:
            mask = np.isin(classes, classes_to_detect)
            class_idxs = np.nonzero(mask)
            outputs = discriminate(outputs, class_idxs)
            out_image = output_image(cfg, img, outputs)
            num_of_instances = len(outputs['instances']) # сколько объектов найдено
            what_instances = outputs['instances'].pred_classes # какие классы найдены
        st.image(out_image, caption='Загруженное и обработанное изображение', use_column_width=True)
        return num_of_instances, what_instances    

    # Streamlit initialization
    st.markdown('''<h1 style='text-align: center; color: #9F2B68;'
            >Инстанс-сегментация изображений</h1>''', 
            unsafe_allow_html=True)

    st.markdown('''<h3 style='text-align: center; color: grey;'
            >Instance segmentation</h3>''', 
            unsafe_allow_html=True)

    st.write("""
        Данное приложение позволяет познакомиться с основами сегментации изображений. 
        \nДля этого используется модель *mask_rcnn_R_50_FPN_3x*, обученная распознавать и выделять на фото 80 классов изображений.
        \n**Используемые библиотеки:** [cv2](https://opencv.org/),[Detectron2](https://detectron2.readthedocs.io/en/latest/), [Streamlit](https://docs.streamlit.io/library/get-started).

        \n**Полезно почитать:** [О библиотеке Detectron2](https://github.com/facebookresearch/detectron2), [Семантическая и инстанс-сегменация](https://neurohive.io/ru/osnovy-data-science/semantic-segmention/).
        \nДанные подготовили сотрудники ЛИА РАНХиГС.
        """)

    # ----------------Pipeline description----------------#
    img_pipeline = Image.open('Pipeline_for_instance_segmentation.png') #
    st.image(img_pipeline, use_column_width='auto', caption='Общий пайплайн для приложения') #width=450

    pipeline_bar = st.expander("Пайплайн микросервиса:")
    pipeline_bar.markdown(
    """
    \n**Этапы:**
    \n(зелёным обозначены этапы, корректировка которых доступна студенту, красным - то этапы, что предобработаны и скорректированы сотрудником лаборатории)
    \n1. Выбор предобученной модели:
    \n*заранее проведённые этапы*:
    \n* сбор, разметка изображений: для обучения модели mask_rcnn_R_50_FPN_3x (библиотека Detectron2) было собрано и размечено более 300 тыс. изображений
    \n* обучение и валидация модели: таким образом модель mask_rcnn_R_50_FPN_3x может распознавать 80 классов объектов на изображении 
    \n* сохранение модели
    \n2. Настройка гиперпараметров модели: к ним относятся, например, порог определения объектов моделью
    \n3. Написание функций обработки изображения: функции преобразования загруженного изображения в типа данных, доступный для приёма моделью, функции выведения bounding box и результата сегментации на исходном изображении
    \n4. Загрузка нового изображения: с использованием библиотеки streamlit
    \n5. Проверка результата: визуально
    \n6. Корректировки параметров: если результат неудовлетворительный, проводится корректировка гиперпараметров и функций из п.2 и п.3
    \n7. Оформление микросервиса Streamlit, выгрузка на сервер: проводится сотрудником лаборатории, используется студентами РАНХиГС
    """)
    
    # ----------------Зона ознакомления----------------
    st.markdown('''<h2 style='text-align: center; color: black;'
            >Инстанс-сегментация изображений</h2>''', 
            unsafe_allow_html=True)

    info_bar = st.expander("Что такое сегментация изображений?")
    info_bar.markdown(
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
        
        """ )

    st.markdown("""
        Вам необходимо выбрать классы для сегментации, а так же загрузить само изображение.
    \nЕсли не указывать определённый класс, модель автоматически распознает и выделит на фото все найденные классы из 80 имеющихся.
    """)
    exp_bar = st.expander("Для справки: классы объектов, которые распознаёт mask_rcnn_R_50_FPN_3x")
    exp_bar.code(classes)

    ## ----------Выбор классов для моделей----------
    select_classes = st.multiselect("Выберите классы для определения на изображении:", classes, ['person'])
    
    # mask = np.isin(classes, classes_to_detect)
    # class_idxs = np.nonzero(mask)

    # Место для изображения
    img_placeholder = st.empty()

    uploaded_image = st.file_uploader("Загрузите изображение ниже:", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        image_segmentation(uploaded_image, select_classes)
       
    # ----------------ЗОНА ЛАБОРАТОРНОЙ РАБОТЫ----------------
    st.markdown('''<h2 style='text-align: center; color: black;'>
                Лабораторная работа
                </h2>''', unsafe_allow_html=True)  
    # ----------------ЗАДАНИЕ 1----------------            
    st.markdown('''<h3 style='text-align: Left; color: black;'> 
            Задание 1
            </h3>''', unsafe_allow_html=True)   

    st.markdown(''' \n*Подсчёт количества людей на фото* 
    \nПриходилось ли вам когда-нибудь быстро оценивать, сколько людей пришло на мероприятие, 
    много ли посетителей в торговом зале или какова длина очереди? 
    \nДавайте посмотрим, насколько быстро 
    и качественно это сделаете вы и насколько быстро и качественно это сделает наша 
    нейросеть.''')
           
    st.markdown(''' \n*1. Самостоятельный подсчёт:*
    \nПеред вами появится изображение большого количества людей. Ваша задача - посчитать на нём каждого человека.
    \n**Порядок действий:**
    \n1) Когда будете готовы, - нажмите на "галочку" "Начать задание 1.1". Одновременно с этим загрузится и само фото. Посчитайте количество человек на нём.
    \n2) Когда вы посчитате всех людей, - впишите полученное число в соответствующую строку и нажмите Enter.
    \n3) Запишите ваши результаты.
    ''')
 #---таймер---   
    if st.checkbox('Начать задание 1.1'):
        st.image("photos/crowd_people.jpg", use_column_width='auto', caption=f'Загруженное изображение "crowd_people.jpg"')
        # start_time_lab_1_student = time.time() # начало счётчика    
        if 'student_answer_lab1' not in st.session_state:
            st.session_state.student_answer_lab1 = time.time() # начало счётчика  
        student_answer_lab1 = st.number_input('Сколько людей на фото?', min_value=0)
        if student_answer_lab1>0:
            st.write("Вы нашли", student_answer_lab1, "людей из 37. Определение заняло", round((time.time() - st.session_state.student_answer_lab1), 1), "секунд") # конец счётчика                 
        else:
            st.write('**Число людей не фото не введено**')

   
    st.markdown(''' \n*2. Использование модели:*
    \nТеперь посмотрим, как быстро с этой же задачей справится предобученная модель mask_rcnn_R_50_FPN_3x.
    \n**Порядок действий:**
    \n1) Загрузите файл "crowd_people.jpg". После этого начнётся работа модели. 
    \n2) Запишите время определения людей на фото моделью, оцените полноту и качество решения ею этой задачи.
    \n3) Сравните со своими результатами. Как думаете, сколько времени бы вам понадобилось, если бы изображение было в 3, 10, 100 раз больше предложенного?
    ''')
    lab_img_1 = st.file_uploader('Выберите файл "crowd_people.jpg" и загрузите его:', type=['jpg', 'jpeg', 'png'])
    if lab_img_1 is not None: 
        start_time_lab_1_model = time.time() # начало счётчика
        num_of_instances_lab_img_1, what_instances_lab_img_1 = image_segmentation(lab_img_1, classes_to_detect='person') #%time
        st.write('Найдено людей:', num_of_instances_lab_img_1, ". Определение заняло", round((time.time() - start_time_lab_1_model), 1), "секунд") # конец счётчика   

    # ----------------ЗАДАНИЕ 2---------------- 
    st.markdown('''<h3 style='text-align: Left; color: black;'>
            Задание 2
            </h3>''', unsafe_allow_html=True) # Задание 2
    st.markdown(''' \n*Подсчёт различных видов похожих друг на друга объектов* 
    \nВ этом разделе мы немного усложним задание: вам будет предложено посчитать определенный сорт фруктов среди других объектов, похожих на них. 
    \nВаши результаты сравним с результатами нейронной сети.''')
    st.markdown(''' \n*1. Самостоятельный подсчёт:*
    \nВам будут предложены 2 изображения. Ваша задача - посчитать на нём количество определённых фруктов, в зависимости от задания ниже.
    \n**Порядок действий:**
    \n1) Когда будете готовы, - нажмите на "галочку" "Начать задание 2.1". Одновременно с этим загрузится и само фото. Посчитайте количество **бананов** на нём.
    \n2) Когда посчитаете, - впишите полученное число в соответствующую строку и нажмите Enter.
    \n3) Запишите ваши результаты.

    \n4) Когда будете готовы ко второму заданию, - нажмите на "галочку" "Начать задание 2.2". Загрузится второе изображение. Посчитайте количество **всех яблок** на нём.
    \n5) Когда посчитаете, - впишите полученное число в соответствующую строку и нажмите Enter.
    \n6) Запишите ваши результаты.
    ''')
    if st.checkbox('Начать задание 2.1'):
        st.image("photos/bananas_photo.png", use_column_width='auto', caption=f'Загруженное изображение "bananas_photo.png"')   
        if 'student_answer_lab2_1' not in st.session_state:
            st.session_state.student_answer_lab2_1 = time.time() # начало счётчика  
        student_answer_lab2_1 = st.number_input('Сколько бананов изображено?', min_value=0)
        if student_answer_lab2_1>0:
            st.write("Количество бананов, что вы нашли:", int(student_answer_lab2_1), "из 14. Определение заняло", round((time.time() - st.session_state.student_answer_lab2_1), 1), "секунд") # конец счётчика                 
        else:
            st.write('**Введите количество бананов на фото**')

    if st.checkbox('Начать задание 2.2'):
        st.image("photos/apples_photo.png", use_column_width='auto', caption=f'Загруженное изображение "apples_photo.jpg"')  
        if 'student_answer_lab2_2' not in st.session_state:
            st.session_state.student_answer_lab2_2 = time.time() # начало счётчика  
        student_answer_lab2_2 = st.number_input('Сколько всего яблок на изображении?', min_value=0)
        if student_answer_lab2_2>0:
            st.write("Количество яблок, что вы нашли:", int(student_answer_lab2_2), "из 22. Определение заняло", round((time.time() - st.session_state.student_answer_lab2_2), 1), "секунд") # конец счётчика                 
        else:
            st.write('**Введите количество бананов на фото**')

    
    st.markdown(''' \n*2. Использование модели:*
    \nТеперь посмотрим, как быстро с этой же задачей справится всё та же нейронная сеть mask_rcnn_R_50_FPN_3x.
    \n**Порядок действий:**
    \n1) Загрузите файл "bananas_photo.png". 
    \n2) Выберете для подсчёта класс 'banana' в окне ниже.
    \n3) Нажмите на кнопку 'Сегментация моделью': после этого начнётся работа модели. 
    \n4) Запишите время определения бананов на фото моделью, оцените полноту и качество поиска.
    \n5) Вернитесь на этап 1): загрузите файл "apples_photo.png". 
    \n6) Выберете для подсчёта класс 'apple'.
    \n7) Нажмите на кнопку 'Сегментация моделью'.
    \n8) Запишите время определения моделью яблок на фото, оцените полноту и качество поиска.
    \n9) Сравните со своими результатами. Как думаете, сколько времени бы вам понадобилось, если бы изображения были в 5 раз больше предложенных?
    ''')

    lab_img_2 = st.file_uploader('Выберите изображение и загрузите его:', type=['jpg', 'jpeg', 'png'])
    if lab_img_2 is not None: 
        st.image(lab_img_2, use_column_width='auto', caption=f'Загруженное изображение {lab_img_2.name}')

    lab_2_select_classes = st.multiselect("Выберите классы для определения:", ['banana', 'apple']) # , 'sandwich', 'orange', 'broccoli', 'carrot'
    if st.button('Сегментация моделью'):
        if lab_img_2 and lab_2_select_classes is not None:
            start_time = time.time()
            num_of_instances_lab_img_2, what_instances_lab_img_2 = image_segmentation(lab_img_2, classes_to_detect=lab_2_select_classes)
            st.write(f'Нейросеть нашла', num_of_instances_lab_img_2, 'объекта класса', lab_2_select_classes[0])
            st.write("Определение заняло", round((time.time() - start_time), 1), "секунд")
            

    st.markdown('''<h3 style='text-align: Left; color: black;'>
            Выводы по лабораторной работе:
            </h3>''', unsafe_allow_html=True) # Выводы по лаборатоной работе
    st.markdown(''' \n**1. Сегментации изображений может быть применена к абсолютно разным объектам.**
    \nНа простых примерах вы убедились, что сегментация изображений может быть применима для определения разнообразных объектов: от бытовых предметов до человеческих лиц. 
    Причём изображения одного класса могут отличаться размером, формой, цветом, положением, - и это не составит сложности для хорошо обученной модели.
    \n**2. Количество объектов значительно не влияет на скорость сегментации.**
    \nВ отличие от распознавания объектов человеком, количество распознаваемых объектов не сильно влияет на время и скорость распознавания моделью.
    \n**3. Сегментация изображений - широко используемая, перспективная сфера data science.**
    \nСегментация изображений относится к сфере компьютерного зрения (computer vision), которая работает с любыми видами изображений и оптимизирует работу человека с визуальными данными, упрощая и ускоряя её.''')

    
        

# -----------------ЗАПУСКАЕМ ПРИЛОЖЕНИЕ-----------------
if __name__ == '__main__':
    main()
    
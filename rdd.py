'''
         /=========================================\
  ===##//|_Детектор повреждений дорожного покрытия_|\\##===
         \========================================/

         Распознает различные типы повреждений
         дорожного покрытия. Считает оценку дорожного
         покрытия. Чем больше оценка, тем хуже состояние
         дорожного покрытия.

        d40 - выбоина d20 - обширная трещина d00 - трещина
        d10 - маленькая трещина (d43 - колодец, остальным можно пренебречь)

для запуска >>>python rdd.py
tensorflow version >= 2.3
python version >= 3.7

'''
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL.ExifTags import TAGS
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import matplotlib.pyplot as plt
import os


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def detect(TEST_IMAGE_PATHS, output_directory, detection_graph, category_index):
    IMAGE_SIZE = (12, 8)
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                name = image_path.split('\\')[len(image_path.split('\\')) - 1] # извлекаем им файла из пути
                print(name)
                image = Image.open(image_path)
                # exifdata = image.getexif()  TODO metadata
                # print(exifdata)            # в планах реаллизовать извлечение метаданных
                # for tag_id in exifdata:
                #    tag = TAGS.get(tag_id, tag_id)
                #    data = exifdata.get(tag_id)
                #    if isinstance(data, bytes):
                #        data = data.decode()
                #    print(f"{tag:25}: {data}")

                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run(  # запускаем модель
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(  # визуализация результата на картинке
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    min_score_thresh=0.3,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                score = 0
                koef = 2  # коэффициент учитывающийся при высчитывании оценки.
                # Можно настраивать чувствительность к повреждениям
                for i, b in enumerate(boxes[0]):  # считаем оценку (чем больше, тем хуже)
                    rel_area = round((abs(boxes[0][i][3] - boxes[0][i][1])), 2) * 10
                    if scores[0][i] >= 0.4: # если модель уверена > чем на 40% то мы учитываем то, что она распознала
                        if classes[0][i] < 5: # значимые повреждения
                            score += classes[0][i] * rel_area / koef
                        else:
                            score += rel_area # незначительные (можно игнорировать)
                        #print(classes[0][i], " ", rel_area)
                score = round(score, 1)
                print("Score: ", score)
                print("===================")

                plt.figure(figsize=IMAGE_SIZE)  # сохраняем обработанную картинку
                plt.suptitle('Score: ' + str(score), fontsize=14, fontweight='bold')
                plt.imshow(image_np)
                plt.savefig(output_directory + '\\' + name, bbox_inches="tight")


def main():
    PATH_TO_PTM = 'resnet.pb'  # путь к обученой модели

    PATH_TO_LABELS = 'label_map.pbtxt'  # путь к типам повреждений #d40 - выбоина d20 - обширная трещина
    # d00 - трещина d10 - маленькая трещина (d43 - колодец, остальным можно пренебречь)

    NUM_CLASSES = 7

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(PATH_TO_PTM, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    input_directory = 'input_data'  # путь к папке с входными данными
    output_directory = 'output_data' # путь к папке с выходными данными

    TEST_IMAGE_PATHS = []  # сюда запишем пути к картинкам

    for filename in os.listdir(input_directory):
        TEST_IMAGE_PATHS.append(input_directory + '\\' + filename);
    detect(TEST_IMAGE_PATHS, output_directory, detection_graph, category_index)

main()

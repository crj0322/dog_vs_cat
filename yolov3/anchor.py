import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def read_boxes(file_list, xml_path):
    """
    Arguments:
        file_list: xml file names.
        xml_path: xml file path.
    Return:
        wh info.
    """
    boxes = []
    input_wh = np.array([416, 416])
    for file in tqdm(file_list):
        tree =  ET.ElementTree(file=xml_path + os.sep + file)
        
        # image size
        w = []
        h = []
        for elem in tree.iterfind('size/width'):
            w.append(int(elem.text))
        for elem in tree.iterfind('size/height'):
            h.append(int(elem.text))

        assert len(w) == 1 and len(h) == 1
        image_wh = np.array([*w, *h])
        
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        for elem in tree.iterfind('object/bndbox/xmin'):
            xmin.append(int(elem.text))

        for elem in tree.iterfind('object/bndbox/xmax'):
            xmax.append(int(elem.text))

        for elem in tree.iterfind('object/bndbox/ymin'):
            ymin.append(int(elem.text))

        for elem in tree.iterfind('object/bndbox/ymax'):
            ymax.append(int(elem.text))

        xmin = np.array(xmin).reshape((-1, 1))
        xmax = np.array(xmax).reshape((-1, 1))
        ymin = np.array(ymin).reshape((-1, 1))
        ymax = np.array(ymax).reshape((-1, 1))

        wh = np.concatenate([xmax-xmin, ymax-ymin], axis=-1).astype(np.float32)
        wh = wh * input_wh/image_wh
        boxes.append(wh)
        
    return np.concatenate(boxes, axis=0)

def main():
    xml_path = os.path.join('E:\\', 'data', 'The Oxford-IIIT Pet Dataset', 'annotations', 'xmls')
    file_list = os.listdir(xml_path)
    X = read_boxes(file_list, xml_path)
    
    # cluster
    from sklearn.cluster import KMeans
    cluster = KMeans(n_clusters=9, random_state=9, verbose=0)
    y_pred = cluster.fit_predict(X)
    centers = cluster.cluster_centers_.astype(np.int32)
    centers = centers[np.argsort(centers[:,0]*centers[:,1])]
    print(centers)

    fo = open("model/pet_anchors.txt", "w")
    m, n = centers.shape
    for i in range(m):
        for j in range(n):
            text = str(centers[i, j])
            if j != n-1:
                text += ','
            elif i != m-1:
                text += ',  '

            fo.write(text)
    fo.close()

    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()

if __name__ == '__main__':
    main()
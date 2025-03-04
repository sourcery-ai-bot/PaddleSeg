import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
import os
import os.path as osp
from collections import defaultdict
import sys
from datetime import datetime


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # dataset, anns, cats, imgs, imgToAnns, catToImgs, imgNameToId, maxAnnId, maxImgId
        self.dataset = {
            "categories": [],
            "images": [],
            "annotations": [],
            "info": "",
            "licenses": [],
        }  # the complete json
        self.anns = {}
        self.cats = {}
        self.imgs = {}
        self.imgToAnns = defaultdict(list)  # imgToAnns[imgId] = [ann]
        self.catToImgs = defaultdict(list)  # catToImgs[catId] = [imgId]
        self.imgNameToId = defaultdict(list)  # imgNameToId[name] = imgId
        self.maxAnnId = 0
        self.maxImgId = 0
        if annotation_file is not None and osp.exists(annotation_file):
            print("loading annotations into memory...")
            tic = time.time()
            dataset = json.load(open(annotation_file, "r"))
            assert (
                type(dataset) == dict
            ), f"annotation file format {type(dataset)} not supported"
            print("Done (t={:0.2f}s)".format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()
            print(
                f"load coco with {len(self.dataset['images'])} images and {len(self.dataset['annotations'])} annotations."
            )

    def hasImage(self, imageName):
        imgId = self.imgNameToId.get(imageName, None)
        return imgId is not None

    def hasCat(self, catIdx):
        res = self.cats.get(catIdx)
        return res is not None

    def createIndex(self):
        # create index
        print("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgNameToId, imgToAnns, catToImgs, imgNameToId = [
            defaultdict(list) for _ in range(4)
        ]
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann
                self.maxAnnId = max(self.maxAnnId, ann["id"])

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img
                imgNameToId[img["file_name"]] = img["id"]
                try:
                    imgId = int(img["id"])
                    self.maxImgId = max(self.maxImgId, imgId)
                except:
                    pass

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])
        # TODO: read license
        print("index created!")

        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgNameToId = imgNameToId
        self.imgs = imgs
        self.cats = cats

    def setInfo(
            self,
            year: int="",
            version: str="",
            description: str="",
            contributor: str="",
            url: str="",
            date_created: datetime="", ):
        self.dataset["info"] = {
            "year": year,
            "version": version,
            "description": description,
            "contributor": contributor,
            "url": url,
            "date_created": date_created,
        }

    def addCategory(
            self,
            id: int,
            name: str,
            color: list,
            supercategory: str="", ):
        cat = {
            "id": id,
            "name": name,
            "color": color,
            "supercategory": supercategory,
        }
        self.cats[id] = cat
        self.dataset["categories"].append(cat)

    def updateCategory(
            self,
            id: int,
            name: str,
            color: list,
            supercategory: str="", ):
        cat = {
            "id": id,
            "name": name,
            "color": color,
            "supercategory": supercategory,
        }
        self.cats[id] = cat
        for idx in range(len(self.dataset["categories"])):
            if self.dataset["categories"][idx]["id"] == id:
                self.dataset["categories"][idx] = cat

    def addImage(
            self,
            file_name: str,
            width: int,
            height: int,
            id: int=None,
            license: int="",
            flickr_url: str="",
            coco_url: str="",
            date_captured: datetime="", ):
        if self.hasImage(file_name):
            print(f"{file_name}图片已存在")
            return
        if not id:
            self.maxImgId += 1
            id = self.maxImgId
        image = {
            "id": id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": license,
            "flickr_url": flickr_url,
            "coco_url": coco_url,
            "date_captured": date_captured,
        }
        self.dataset["images"].append(image)
        self.imgs[id] = image
        self.imgNameToId[file_name] = id
        return id

    def getBB(self, segmentation):
        x = segmentation[::2]
        y = segmentation[1::2]
        maxx, minx, maxy, miny = max(x), min(x), max(y), min(y)
        return [minx, miny, maxx - minx, maxy - miny]

    def getArea(self, segmentation):
        x = segmentation[::2]
        y = segmentation[1::2]

        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def addAnnotation(
            self,
            image_id: int,
            category_id: int,
            segmentation: list,
            area: float=None,
            id: int=None, ):
        if id is not None and self.anns.get(id, None) is not None:
            print("标签已经存在")
            return
        if not id:
            self.maxAnnId += 1
            id = self.maxAnnId

        ann = {
            "id": id,
            "iscrowd": 0,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [segmentation],
            "area": self.getArea(segmentation),
            "bbox": self.getBB(segmentation),
        }

        self.dataset["annotations"].append(ann)
        self.anns[id] = ann
        self.imgToAnns[image_id].append(ann)
        self.catToImgs[category_id].append(image_id)
        return id

    def delAnnotation(self, annId, imgId):
        if "annotations" in self.dataset:
            for idx, ann in enumerate(self.dataset["annotations"]):
                if ann["id"] == annId:
                    del self.dataset["annotations"][idx]
        if annId in self.anns.keys():
            del self.anns[annId]

        for idx, ann in enumerate(self.imgToAnns[imgId]):
            if ann["id"] == annId:
                del self.imgToAnns[imgId][idx]

    def updateAnnotation(self, id, imgId, segmentation):
        self.anns[id]["segmentation"] = [segmentation]
        self.anns[id]["bbox"] = self.getBB(segmentation)
        self.anns[id]["area"] = self.getArea(segmentation)
        for rec in self.dataset["annotations"]:
            if rec["id"] == id:
                rec = self.anns[id]
                break

        for rec in self.dataset["annotations"]:
            if rec["id"] == id:
                # @todo TODO move into debug codes or controls
                print(
                    "record point : ",
                    rec["segmentation"][0][0],
                    rec["segmentation"][0][1], )
                break

        for rec in self.imgToAnns[imgId]:
            if rec["id"] == id:
                rec["segmentation"] = [segmentation]
                break

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset["info"].items():
            print(f"{key}: {value}")

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            if len(imgIds) != 0:
                lists = [
                    self.imgToAnns[imgId] for imgId in imgIds
                    if imgId in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = (anns if len(catIds) == 0 else
                    [ann for ann in anns if ann["category_id"] in catIds])
            anns = (anns if len(areaRng) == 0 else [
                ann for ann in anns
                if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]
            ])
        return (
            [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
            if iscrowd is not None
            else [ann["id"] for ann in anns]
        )

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        cats = self.dataset["categories"]
        if not len(catNms) == len(supNms) == len(catIds) == 0:
            cats = (cats if len(catNms) == 0 else
                    [cat for cat in cats if cat["name"] in catNms])
            cats = (cats if len(supNms) == 0 else
                    [cat for cat in cats if cat["supercategory"] in supNms])
            cats = (cats if len(catIds) == 0 else
                    [cat for cat in cats if cat["id"] in catIds])
        return [cat["id"] for cat in cats]

    def getImgIds(self, imgIds=[], catIds=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    # def showAnns(self, anns, draw_bbox=False):
    #     """
    #     Display the specified annotations.
    #     :param anns (array of object): annotations to display
    #     :return: None
    #     """
    #     if len(anns) == 0:
    #         return 0
    #     if "segmentation" in anns[0] or "keypoints" in anns[0]:
    #         datasetType = "instances"
    #     elif "caption" in anns[0]:
    #         datasetType = "captions"
    #     else:
    #         raise Exception("datasetType not supported")
    #     if datasetType == "instances":
    #         ax = plt.gca()
    #         ax.set_autoscale_on(False)
    #         polygons = []
    #         color = []
    #         for ann in anns:
    #             c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
    #             if "segmentation" in ann:
    #                 if type(ann["segmentation"]) == list:
    #                     # polygon
    #                     for seg in ann["segmentation"]:
    #                         poly = np.array(seg).reshape((int(len(seg) / 2), 2))
    #                         polygons.append(Polygon(poly))
    #                         color.append(c)
    #                 else:
    #                     # mask
    #                     t = self.imgs[ann["image_id"]]
    #                     if type(ann["segmentation"]["counts"]) == list:
    #                         rle = maskUtils.frPyObjects(
    #                             [ann["segmentation"]], t["height"], t["width"]
    #                         )
    #                     else:
    #                         rle = [ann["segmentation"]]
    #                     m = maskUtils.decode(rle)
    #                     img = np.ones((m.shape[0], m.shape[1], 3))
    #                     if ann["iscrowd"] == 1:
    #                         color_mask = np.array([2.0, 166.0, 101.0]) / 255
    #                     if ann["iscrowd"] == 0:
    #                         color_mask = np.random.random((1, 3)).tolist()[0]
    #                     for i in range(3):
    #                         img[:, :, i] = color_mask[i]
    #                     ax.imshow(np.dstack((img, m * 0.5)))
    #             if "keypoints" in ann and type(ann["keypoints"]) == list:
    #                 # turn skeleton into zero-based index
    #                 sks = np.array(self.loadCats(ann["category_id"])[0]["skeleton"]) - 1
    #                 kp = np.array(ann["keypoints"])
    #                 x = kp[0::3]
    #                 y = kp[1::3]
    #                 v = kp[2::3]
    #                 for sk in sks:
    #                     if np.all(v[sk] > 0):
    #                         plt.plot(x[sk], y[sk], linewidth=3, color=c)
    #                 plt.plot(
    #                     x[v > 0],
    #                     y[v > 0],
    #                     "o",
    #                     markersize=8,
    #                     markerfacecolor=c,
    #                     markeredgecolor="k",
    #                     markeredgewidth=2,
    #                 )
    #                 plt.plot(
    #                     x[v > 1],
    #                     y[v > 1],
    #                     "o",
    #                     markersize=8,
    #                     markerfacecolor=c,
    #                     markeredgecolor=c,
    #                     markeredgewidth=2,
    #                 )
    #
    #             if draw_bbox:
    #                 [bbox_x, bbox_y, bbox_w, bbox_h] = ann["bbox"]
    #                 poly = [
    #                     [bbox_x, bbox_y],
    #                     [bbox_x, bbox_y + bbox_h],
    #                     [bbox_x + bbox_w, bbox_y + bbox_h],
    #                     [bbox_x + bbox_w, bbox_y],
    #                 ]
    #                 np_poly = np.array(poly).reshape((4, 2))
    #                 polygons.append(Polygon(np_poly))
    #                 color.append(c)
    #
    #         p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    #         ax.add_collection(p)
    #         p = PatchCollection(
    #             polygons, facecolor="none", edgecolors=color, linewidths=2
    #         )
    #         ax.add_collection(p)
    #     elif datasetType == "captions":
    #         for ann in anns:
    #             print(ann["caption"])
    #
    # def loadRes(self, resFile):
    #     """
    #     Load result file and return a result api object.
    #     :param   resFile (str)     : file name of result file
    #     :return: res (obj)         : result api object
    #     """
    #     res = COCO()
    #     res.dataset["images"] = [img for img in self.dataset["images"]]
    #
    #     print("Loading and preparing results...")
    #     tic = time.time()
    #     if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
    #         anns = json.load(open(resFile))
    #     elif type(resFile) == np.ndarray:
    #         anns = self.loadNumpyAnnotations(resFile)
    #     else:
    #         anns = resFile
    #     assert type(anns) == list, "results in not an array of objects"
    #     annsImgIds = [ann["image_id"] for ann in anns]
    #     assert set(annsImgIds) == (
    #         set(annsImgIds) & set(self.getImgIds())
    #     ), "Results do not correspond to current coco set"
    #     if "caption" in anns[0]:
    #         imgIds = set([img["id"] for img in res.dataset["images"]]) & set(
    #             [ann["image_id"] for ann in anns]
    #         )
    #         res.dataset["images"] = [
    #             img for img in res.dataset["images"] if img["id"] in imgIds
    #         ]
    #         for id, ann in enumerate(anns):
    #             ann["id"] = id + 1
    #     elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
    #         res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
    #         for id, ann in enumerate(anns):
    #             bb = ann["bbox"]
    #             x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
    #             if not "segmentation" in ann:
    #                 ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
    #             ann["area"] = bb[2] * bb[3]
    #             ann["id"] = id + 1
    #             ann["iscrowd"] = 0
    #     elif "segmentation" in anns[0]:
    #         res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
    #         for id, ann in enumerate(anns):
    #             # now only support compressed RLE format as segmentation results
    #             ann["area"] = maskUtils.area(ann["segmentation"])
    #             if not "bbox" in ann:
    #                 ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
    #             ann["id"] = id + 1
    #             ann["iscrowd"] = 0
    #     elif "keypoints" in anns[0]:
    #         res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
    #         for id, ann in enumerate(anns):
    #             s = ann["keypoints"]
    #             x = s[0::3]
    #             y = s[1::3]
    #             x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
    #             ann["area"] = (x1 - x0) * (y1 - y0)
    #             ann["id"] = id + 1
    #             ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
    #     print("DONE (t={:0.2f}s)".format(time.time() - tic))
    #
    #     res.dataset["annotations"] = anns
    #     res.createIndex()
    #     return res

    def download(self, tarDir=None, imgIds=[]):
        """
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        """
        if tarDir is None:
            print("Please specify target directory")
            return -1
        imgs = self.imgs.values() if len(imgIds) == 0 else self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img["file_name"])
            if not os.path.exists(fname):
                urlretrieve(img["coco_url"], fname)
            print("downloaded {}/{} images (t={:0.1f}s)".format(
                i, N, time.time() - tic))

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print("Converting ndarray to lists...")
        assert type(data) == np.ndarray
        print(data.shape)
        assert data.shape[1] == 7
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print(f"{i}/{N}")
            ann += [{
                "image_id": int(data[i, 0]),
                "bbox": [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                "score": data[i, 5],
                "category_id": int(data[i, 6]),
            }]
        return ann

    # def annToRLE(self, ann):
    #     """
    #     Convert annotation which can be polygons, uncompressed RLE to RLE.
    #     :return: binary mask (numpy 2D array)
    #     """
    #     t = self.imgs[ann["image_id"]]
    #     h, w = t["height"], t["width"]
    #     segm = ann["segmentation"]
    #     if type(segm) == list:
    #         # polygon -- a single object might consist of multiple parts
    #         # we merge all parts into one mask rle code
    #         rles = maskUtils.frPyObjects(segm, h, w)
    #         rle = maskUtils.merge(rles)
    #     elif type(segm["counts"]) == list:
    #         # uncompressed RLE
    #         rle = maskUtils.frPyObjects(segm, h, w)
    #     else:
    #         # rle
    #         rle = ann["segmentation"]
    #     return rle

    # def annToMask(self, ann):
    #     """
    #     Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    #     :return: binary mask (numpy 2D array)
    #     """
    #     rle = self.annToRLE(ann)
    #     m = maskUtils.decode(rle)
    #     return m

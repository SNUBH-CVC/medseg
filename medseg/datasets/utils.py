import os
from collections import defaultdict

from pycocotools.coco import COCO


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


# ann - img 1:1 대응되도록 코드 변경하기,
# ann이 mask_info, skeleton_info 와 같은 key를 가지고 그 안에 데이터를 보관하도록 하기
class PanopticCOCO(COCO):

    def createIndex(self):
        # create index
        print("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns = defaultdict(list)
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        print("index created!")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.imgs = imgs
        self.cats = cats

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
        return list(ids)

    def getImgPath(self, img_id, img_dir):
        img = self.loadImgs(img_id)
        if img:
            return os.path.join(img_dir, img[0]["file_name"])
        return None

    def getSegMapPath(self, img_id, seg_map_dir):
        anns = self.imgToAnns[img_id]
        if anns:
            return os.path.join(seg_map_dir, anns[0]["file_name"])
        return None

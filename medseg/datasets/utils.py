import json


# https://github.com/cocodataset/panopticapi
class PanopticCOCO:

    def __init__(self, annotation_path):
        self.annotation_path = annotation_path
        with open(self.annotation_path, "r") as f:
            self.dataset = json.load(f)
        self.create_index()

    def create_index(self):
        self.imgs = {}
        self.img_to_ann = {}
        for ann in self.dataset["annotations"]:
            self.img_to_ann.update({ann["image_id"]: ann})

        for img in self.dataset["images"]:
            self.imgs.update({img["id"]: img})

    def get_img_ids(self):
        return [i["id"] for i in self.dataset["images"]]

    def load_img(self, img_id):
        return self.imgs[img_id]

    def load_ann(self, img_id):
        return self.img_to_ann[img_id]

import cv2
from typing import List, Tuple
from pvdn.keypoints import Vehicle

class Resize(object):
    def __init__(self, factor=1):
        """
        :param factor: Factor the image width and height are resized by
        """
        self.factor = factor

    def _resize_coords(self, coords:Tuple[int, int], factor):
        x = int(coords[0] * factor + 0.5)
        y = int(coords[1] * factor + 0.5)
        return (x,y)

    def __call__(
        self,
        img,
        vehicles: List[Vehicle]
    ):
        """
        Resize the image height and width by the provided factor and move keypoints in vehicles respectively.
        :param img: Image of shape numpy.ndarray([h, w])
        :param vehicles: List of PVDN vehicle objects
        """
        # resize image
        resized_img = cv2.resize(
            img, (0, 0), fx=self.factor, fy=self.factor
        )
        # move keypoints
        for vehicle in vehicles:
            vehicle.position = self._resize_coords(vehicle.position, self.factor)
            for instance in vehicle.instances:
                instance.position = self._resize_coords(instance.position, self.factor)

        return resized_img, vehicles
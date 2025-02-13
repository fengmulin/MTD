from concern.config import Configurable


class DataProcess(Configurable):
    r'''Processes of data dict.
    '''

    def __call__(self, images, polygons):
        return self.process(images, polygons)

    def process(self, images, polygons):
        raise NotImplementedError

    def render_constant(self, canvas, xmin, xmax, ymin, ymax, value=1, shrink=0):
        def shrink_rect(xmin, xmax, ratio):
            center = (xmin + xmax) / 2
            width = center - xmin
            return int(center - width * ratio + 0.5), int(center + width * ratio + 0.5)

        if shrink > 0:
            xmin, xmax = shrink_rect(xmin, xmax, shrink)
            ymin, ymax = shrink_rect(ymin, ymax, shrink)

        canvas[int(ymin+0.5):int(ymax+0.5)+1, int(xmin+0.5):int(xmax+0.5)+1] = value
        return canvas
    

class DataProcess2(Configurable):
    r'''Processes of data dict.
    '''

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        raise NotImplementedError

    def render_constant(self, canvas, xmin, xmax, ymin, ymax, value=1, shrink=0):
        def shrink_rect(xmin, xmax, ratio):
            center = (xmin + xmax) / 2
            width = center - xmin
            return int(center - width * ratio + 0.5), int(center + width * ratio + 0.5)

        if shrink > 0:
            xmin, xmax = shrink_rect(xmin, xmax, shrink)
            ymin, ymax = shrink_rect(ymin, ymax, shrink)

        canvas[int(ymin+0.5):int(ymax+0.5)+1, int(xmin+0.5):int(xmax+0.5)+1] = value
        return canvas

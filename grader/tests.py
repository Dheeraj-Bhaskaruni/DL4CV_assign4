"""
EDIT THIS FILE AT YOUR OWN RISK!
It will not ship with your code, editing it will only change the test cases locally, and might make you fail our
remote tests.
"""
import torch
from .grader import Grader, Case, MultiCase

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_data(dataset, num_workers=0, batch_size=128):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)


class DiceLossGrader(Grader):
    """DiceLoss Grader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.module.DiceLoss()
        self.model.eval()

    @MultiCase(score=30, shape=[(2**i, 2**i) for i in range(10)])
    def test_dice(self, shape):
        x = torch.zeros(1, 3, *shape)
        x[:,0] = 1
        y = torch.zeros(1, 3, *shape)
        y[:,0] = 1
        o = self.model(x, y)
        assert o == -1, '-1 value expected, got (%d)'%(o)
        x = torch.zeros(1, 3, *shape)
        x[:,1] = 1
        y = torch.zeros(1, 3, *shape)
        y[:,0] = 1
        o = self.model(x, y)
        assert o == 0, '0 value expected, got (%d)'%(o)


class FCNGrader(Grader):
    """FCN Grader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.module.FCN()
        self.model.eval()

    @MultiCase(score=30, shape=[(2**i, 2**i) for i in range(10)] + [(2**(5-i), 2**i) for i in range(5)])
    def test_shape(self, shape):
        """Shape"""
        v = torch.zeros(1, 3, *shape)
        o = self.model(v)
        assert o.shape[2:] == v.shape[2:] and o.size(1) == 5 and o.size(0) == 1,\
            'Output shape (1, 5, %d, %d) expected, got (%d, %d, %d, %d)' % (v.size(2), v.size(3), o.size(0), o.size(1),
                                                                            o.size(2), o.size(3))


class TrainedFCNGrader(Grader):
    """Trained FCN Grader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls = self.module.load_model('fcn')
        cls.eval()
        cls = cls.to(device)
        self.c = self.module.utils.ConfusionMatrix()
        for img, label in load_data(self.module.utils.DenseSuperTuxDataset('dense_data/valid')):
            self.c.add(cls(img.to(device)).argmax(1), label.to(device))

    @Case(score=20)
    def test_global_accuracy(self, min_val=0.70, max_val=0.85):
        """Global accuracy"""
        v = self.c.global_accuracy
        return max(min(v, max_val) - min_val, 0) / (max_val - min_val), '%0.3f' % v

    @Case(score=20)
    def test_iou(self, min_val=0.30, max_val=0.55):
        """Intersection over Union"""
        v = self.c.iou
        return max(min(v, max_val) - min_val, 0) / (max_val - min_val), '%0.3f' % v

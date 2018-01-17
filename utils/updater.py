import copy
import six

from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function, variable
from chainer.training.updater import StandardUpdater

class SubDivisionUpdater(StandardUpdater):


    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
        subdivisions=1, device=None, loss_func=None):
        super(SubDivisionUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            device=device,
            loss_func=loss_func,
        )
        self._batchsize = self._iterators['main'].batch_size
        self._subdivisions = subdivisions
        self._n = int(self._batchsize / self._subdivisions)
        assert self._batchsize % self._subdivisions == 0, (self._batchsize, self._subdivisions)

    def update_core(self):
        batch = self._iterators['main'].next()
        #print(self._n)
        in_arrays_list = []
        for i in range(self._n):
            #in_arrays_list.append(self.converter(batch[i::self._subdivisions], self.device))
            in_arrays_list.append(self.converter(batch, self.device))
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target
        loss_func.cleargrads()

        losses=[]

        for i, in_arrays in enumerate(in_arrays_list):
            if isinstance(in_arrays, tuple):
                in_vars = list(variable.Variable(x) for x in in_arrays)
                loss = loss_func(*in_vars)
                losses.append(loss)
            elif isinstance(in_arrays, dict):
                in_vars = {key: variable.Variable(x) for key, x in six.iteritems(in_arrays)}
                loss = loss_func(in_vars)
                losses.append(loss)
            else:
                print(type(in_arrays))
            loss.backward()
        
        optimizer.update()
import copy
import six

from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function, variable
from chainer.training.updater import StandardUpdater
from chainer import reporter

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
        for i in range(self._subdivisions):
            in_arrays_list.append(self.converter(batch[i::self._subdivisions], self.device))
            #in_arrays_list.append(self.converter(batch, self.device))
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
        # minibatch average
        if isinstance(loss, dict):
            avg_loss = {k: 0. for k in losses[0].keys()}
            for loss in losses:
                for k, v in loss.items():
                    avg_loss[k] += v
            #avg_loss = {k: v / float(self._batchsize) for k, v in avg_loss.items()}
            avg_loss = {k: v / float(len(losses)) for k, v in avg_loss.items()}
            #avg_loss = {k: v for k, v in avg_loss.items()}

            # report all the loss values
            for k, v in avg_loss.items():
                reporter.report({k: v}, loss_func)
            reporter.report({'loss': sum(list(avg_loss.values()))}, loss_func)
        else:
            avg_loss = 0.
            for loss in losses:
                avg_loss += loss
            #avg_loss /= float(self._batchsize)
            reporter.report({'loss': avg_loss}, loss_func)
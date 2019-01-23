import numpy as np
import time
import lmdb
import sys
import numpy as np

caffe_root = '/home/jiaxin/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe


lmdb_env = lmdb.open('../0.5hz_new/lmdb/train_label')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

print lmdb_env.stat()
line_number = lmdb_env.stat()['entries']
# data_array = np.zeros((line_number, 2,1081))

counter = 0
# for key, value in lmdb_cursor:
#     datum.ParseFromString(value)
#     label = datum.label
#     data = caffe.io.datum_to_array(datum)
#
#     for d in data:
#         print np.array(d)
#         data_array[counter, :] = np.array(d)
#         # print('mean: {0}'.format(np.mean(d)))
#         # print('std: {0}'.format(np.std(d)))
#     counter += 1


# print('line: {0}'.format(counter))
#
# print('dx:')
# print np.mean(data_array[:,0])
# print np.std(data_array[:,0])
#
# print('dy')
# print np.mean(data_array[:,1])
# print np.std(data_array[:,1])
#
# print('dtheta')
# print np.mean(data_array[:,2])
# print np.std(data_array[:,2])
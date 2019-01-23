import numpy as np
import time
import lmdb
import sys
import math

caffe_root = '/data/jiaxin/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe


COMMIT_LENGTH = 10000


class Post_process_lmdb:
    '''The construction of this class requires the path to the folder that stores train & val & test data/label.'''
    def __init__(self, input_path):
        self.m_input_path = input_path

    def __enter__(self):
        return self

    def __del__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_noise_to_datum(self, raw_datum, mean, std):
        print 'Noise should be added in npz2lmdb.'
        # datum = caffe.proto.caffe_pb2.Datum()
        # datum.ParseFromString(raw_datum)
        #
        # flat_data = np.fromiter(datum.float_data, dtype=np.float32)


    def shuffle_lmdb_data_only(self, data_type_str):
        print "Begin shuffle lmdb"

        input_data_env = lmdb.open('lmdb/'+data_type_str+'_data', readonly=True)
        input_data_txn = input_data_env.begin(buffers=True)
        input_data_cursor = input_data_txn.cursor()

        shuffle_data_env = lmdb.open('lmdb_post/'+data_type_str+'_data', map_size=300*1024*1024*1024)
        shuffle_data_txn = shuffle_data_env.begin(write=True, buffers=True)


        # get number of entries in lmdb
        line_number = input_data_env.stat()['entries']
        permutated_idx_list = np.random.permutation(line_number)


        for i, (key, value) in enumerate(input_data_cursor):
            permutated_idx = permutated_idx_list[i]

            #raw_datum_data = input_data_txn.get('{:010}'.format(permutated_idx))
            shuffle_data_txn.put('{:010}'.format(permutated_idx), value)


            if i>0 and i%COMMIT_LENGTH==0:
                # commit data to disk for each batch
                shuffle_data_txn.commit()
                shuffle_data_txn = shuffle_data_env.begin(write=True, buffers=True)

        shuffle_data_txn.commit()
        input_data_env.close()
        shuffle_data_env.close()

        print "Done shuffling lmdb, data only."


    def shuffle_lmdb(self, data_type_str):
        print "Begin shuffle lmdb"

        input_data_env = lmdb.open('lmdb/'+data_type_str+'_data', readonly=True)
        input_data_txn = input_data_env.begin(buffers=True)
        input_data_cursor = input_data_txn.cursor()
        input_label_env = lmdb.open('lmdb/'+data_type_str+'_label', readonly=True)
        input_label_txn = input_label_env.begin(buffers=True)
        input_label_cursor = input_label_txn.cursor()

        shuffle_data_env = lmdb.open('lmdb_post/'+data_type_str+'_data', map_size=300*1024*1024*1024)
        shuffle_data_txn = shuffle_data_env.begin(write=True, buffers=True)
        shuffle_label_env = lmdb.open('lmdb_post/' + data_type_str + '_label', map_size=60*1024*1024*1024)
        shuffle_label_txn = shuffle_label_env.begin(write=True, buffers=True)


        # get number of entries in lmdb
        line_number = input_data_env.stat()['entries']
        permutated_idx_list = np.random.permutation(line_number)

        label_datum_list = []
        for i, (key, value) in enumerate(input_label_cursor):
            label_datum_list.append(value)

        for i, (key, value) in enumerate(input_data_cursor):
            permutated_idx = permutated_idx_list[i]

            #raw_datum_data = input_data_txn.get('{:010}'.format(permutated_idx))
            shuffle_data_txn.put('{:010}'.format(permutated_idx), value)

            #raw_datum_label = input_label_txn.get('{:010}'.format(permutated_idx))
            shuffle_label_txn.put('{:010}'.format(permutated_idx), label_datum_list[i])

            if i>0 and i%COMMIT_LENGTH==0:
                # commit data to disk for each batch
                shuffle_data_txn.commit()
                shuffle_data_txn = shuffle_data_env.begin(write=True, buffers=True)
                shuffle_label_txn.commit()
                shuffle_label_txn = shuffle_label_env.begin(write=True, buffers=True)

        shuffle_data_txn.commit()
        shuffle_label_txn.commit()
        input_data_env.close()
        input_label_env.close()
        shuffle_data_env.close()
        shuffle_label_env.close()

        print "Done shuffling lmdb"




if __name__=="__main__":
    print('Begin shuffle data.')

    t0 = time.time()
    with Post_process_lmdb('lmdb/') as post_processing:
        #post_processing.shuffle_lmdb('train')
        post_processing.shuffle_lmdb_data_only('train')

    print('Time consumed for shuffling: {0}'.format(time.time()-t0))
import numpy as np
import time
import lmdb
import sys
import math

caffe_root = '/home/jiaxin/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe


COMMIT_LENGTH = 10000

def INPI(angle):
    if angle > math.pi:
        return angle - 2 * math.pi
    elif angle < -1 * math.pi:
        return angle + 2 * math.pi
    else:
        return angle

class Npz2lmdb_Invert:
    """Normalization & noise addition of scan will be done here."""
    def __init__(self, npz_filename, original_frequency):
        self.m_original_frequency = original_frequency

        self.m_npz_filename = npz_filename
        npz_file = np.load(npz_filename)

        self.m_scan = npz_file['scan']
        self.m_reference_absolute = npz_file['reference_absolute']
        self.m_scan_noisy = np.copy(self.m_scan)

        self.m_train_line_counter = 0
        self.m_val_line_counter = 0
        self.m_test_line_counter = 0

        # scan normalization parameters
        self.m_scan_mean = 0
        self.m_scan_std = 0

        self.m_label_array = []
        self.m_label_dx_mean = 0
        self.m_label_dx_std = 0
        self.m_label_dy_mean = 0
        self.m_label_dy_std = 0
        self.m_label_dtheta_mean = 0
        self.m_label_dtheta_std = 0

        print("Npz2lmdb_Invert ----------------")

    def __enter__(self):
        return self

    def __del__(self):
        del self.m_scan
        del self.m_reference_absolute

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_noise_to_scan(self):
        # hokuyo utm_30lx have the accuracy of: 0.1 to 10m:+-30mm, 10 to 30m:+-50mm
        noise_mean = 0
        noise_std = 0.05 / 3
        for line in self.m_scan_noisy:
            noise = noise_std * np.random.randn(line.shape[0]) + noise_mean
            line += noise

    def normalize_scan(self):
        self.m_scan_mean = np.mean(self.m_scan_noisy)
        self.m_scan_std = np.std(self.m_scan_noisy)
        print('scan mean: {0}'.format(self.m_scan_mean))
        print('scan std: {0}'.format(self.m_scan_std))

        self.m_scan -= self.m_scan_mean
        self.m_scan /= self.m_scan_std
        self.m_scan_noisy -= self.m_scan_mean
        self.m_scan_noisy /= self.m_scan_std

    def insert_data_into_lmdb(self, frequency, path, is_add_to_label_stat=True):
        train_length = int(math.floor(self.m_scan.shape[0]*0.8))
        val_length = int(math.floor(self.m_scan.shape[0]*0.1))
        test_length = int(math.floor(self.m_scan.shape[0]*0.1))
        # train_length = 24 * 10000
        # val_length = 3 * 10000
        # test_length = 3 * 10000

        print('Size of scan (MByte): {0}'.format(float(self.m_scan.nbytes)/1000000))

        interval = int(round(self.m_original_frequency / frequency))

        # determine map_size of lmdb
        if 10*self.m_scan.nbytes>10000000:
            map_size_data = 20*self.m_scan.nbytes * 2
        else:
            map_size_data = 20000000
        if 100*self.m_reference_absolute.nbytes>10000000:
            map_size_label = 200*self.m_reference_absolute.nbytes * 2
        else:
            map_size_label = 20000000


        # insert into the train lmdb, including data & label ==================================
        train_data_env = lmdb.open(path+'train_data', map_size=map_size_data)
        train_data_txn = train_data_env.begin(write=True, buffers=True)

        train_label_env = lmdb.open(path+'train_label', map_size=map_size_label)
        train_label_txn = train_label_env.begin(write=True, buffers=True)

        for i in range(train_length-interval):
            # positive
            one_data = np.zeros((1,2,1081), dtype=np.float32)
            one_data[0,0,:] = self.m_scan_noisy[i,:]
            one_data[0,1,:] = self.m_scan_noisy[i+interval,:]

            datum = caffe.io.array_to_datum(one_data.astype(np.float))
            str_id = '{:010}'.format(self.m_train_line_counter+i*2)
            train_data_txn.put(str_id, datum.SerializeToString())

            # invert
            one_data = np.zeros((1, 2, 1081), dtype=np.float32)
            one_data[0, 1, :] = self.m_scan_noisy[i, :]
            one_data[0, 0, :] = self.m_scan_noisy[i + interval, :]

            datum = caffe.io.array_to_datum(one_data.astype(np.float))
            str_id = '{:010}'.format(self.m_train_line_counter + i*2 + 1)
            train_data_txn.put(str_id, datum.SerializeToString())



            # calculate the relative dx, dy, dtheta ----begin
            idx_prev = i
            idx_curr = i+interval
            yaw_prev_ref = self.m_reference_absolute[idx_prev, 2]
            x_prev_curr_ref = self.m_reference_absolute[idx_curr, 0] - self.m_reference_absolute[idx_prev, 0]
            y_prev_curr_ref = self.m_reference_absolute[idx_curr, 1] - self.m_reference_absolute[idx_prev, 1]
            theta_prev_curr_ref = self.m_reference_absolute[idx_curr, 2] - self.m_reference_absolute[idx_prev, 2]

            dx_ref = x_prev_curr_ref * math.cos(yaw_prev_ref) + y_prev_curr_ref * math.sin(yaw_prev_ref)
            dy_ref = y_prev_curr_ref * math.cos(yaw_prev_ref) - x_prev_curr_ref * math.sin(yaw_prev_ref)
            dtheta_ref = theta_prev_curr_ref
            # calculate the relative dx, dy, dtheta ----end


            # positive
            one_label = np.zeros((1,1,3), dtype=np.float32)
            one_label[0,0,0] = dx_ref
            one_label[0,0,1] = dy_ref
            one_label[0,0,2] = INPI(dtheta_ref)
            self.m_label_array.append([dx_ref, dy_ref, INPI(dtheta_ref)])

            datum = caffe.io.array_to_datum(one_label.astype(np.float))
            str_id = '{:010}'.format(self.m_train_line_counter+i*2)
            train_label_txn.put(str_id, datum.SerializeToString())

            # invert
            one_label = np.zeros((1, 1, 3), dtype=np.float32)
            one_label[0, 0, 0] = -dx_ref*math.cos(dtheta_ref) - dy_ref*math.sin(dtheta_ref)
            one_label[0, 0, 1] =  dx_ref*math.sin(dtheta_ref) - dy_ref*math.cos(dtheta_ref)
            one_label[0, 0, 2] = INPI(-dtheta_ref)
            self.m_label_array.append([one_label[0, 0, 0], one_label[0, 0, 1], one_label[0, 0, 2] ])

            datum = caffe.io.array_to_datum(one_label.astype(np.float))
            str_id = '{:010}'.format(self.m_train_line_counter + i*2+1)
            train_label_txn.put(str_id, datum.SerializeToString())


            if i>0 and i%COMMIT_LENGTH==0:
                # commit data to disk every 1000 lines
                train_data_txn.commit()
                train_label_txn.commit()
                train_data_txn = train_data_env.begin(write=True, buffers=True)
                train_label_txn = train_label_env.begin(write=True, buffers=True)

        train_data_txn.commit()
        train_label_txn.commit()
        train_data_env.close()
        train_label_env.close()



        # insert into the val lmdb, including data & label ==================================
        val_data_env = lmdb.open(path+'val_data', map_size=map_size_data)
        val_data_txn = val_data_env.begin(write=True)

        val_label_env = lmdb.open(path+'val_label', map_size=map_size_label)
        val_label_txn = val_label_env.begin(write=True)

        for i in range(val_length - interval):
            # positive
            one_data = np.zeros((1, 2, 1081), dtype=np.float32)
            one_data[0, 0, :] = self.m_scan[i+train_length, :]
            one_data[0, 1, :] = self.m_scan[i+train_length + interval, :]

            datum = caffe.io.array_to_datum(one_data.astype(np.float))
            str_id = '{:010}'.format(self.m_val_line_counter+i*2)
            val_data_txn.put(str_id, datum.SerializeToString())

            # invert
            one_data = np.zeros((1, 2, 1081), dtype=np.float32)
            one_data[0, 1, :] = self.m_scan[i + train_length, :]
            one_data[0, 0, :] = self.m_scan[i + train_length + interval, :]

            datum = caffe.io.array_to_datum(one_data.astype(np.float))
            str_id = '{:010}'.format(self.m_val_line_counter + i*2+1)
            val_data_txn.put(str_id, datum.SerializeToString())


            # calculate the relative dx, dy, dtheta ----begin
            idx_prev = i+train_length
            idx_curr = i+train_length + interval
            yaw_prev_ref = self.m_reference_absolute[idx_prev, 2]
            x_prev_curr_ref = self.m_reference_absolute[idx_curr, 0] - self.m_reference_absolute[idx_prev, 0]
            y_prev_curr_ref = self.m_reference_absolute[idx_curr, 1] - self.m_reference_absolute[idx_prev, 1]
            theta_prev_curr_ref = self.m_reference_absolute[idx_curr, 2] - self.m_reference_absolute[idx_prev, 2]

            dx_ref = x_prev_curr_ref * math.cos(yaw_prev_ref) + y_prev_curr_ref * math.sin(yaw_prev_ref)
            dy_ref = y_prev_curr_ref * math.cos(yaw_prev_ref) - x_prev_curr_ref * math.sin(yaw_prev_ref)
            dtheta_ref = theta_prev_curr_ref
            # calculate the relative dx, dy, dtheta ----end

            # positive
            one_label = np.zeros((1, 1, 3), dtype=np.float32)
            one_label[0, 0, 0] = dx_ref
            one_label[0, 0, 1] = dy_ref
            one_label[0, 0, 2] = INPI(dtheta_ref)

            datum = caffe.io.array_to_datum(one_label.astype(np.float))
            str_id = '{:010}'.format(self.m_val_line_counter+i*2)
            val_label_txn.put(str_id, datum.SerializeToString())

            # invert
            one_label = np.zeros((1, 1, 3), dtype=np.float32)
            one_label[0, 0, 0] = -dx_ref * math.cos(dtheta_ref) - dy_ref * math.sin(dtheta_ref)
            one_label[0, 0, 1] = dx_ref * math.sin(dtheta_ref) - dy_ref * math.cos(dtheta_ref)
            one_label[0, 0, 2] = INPI(-dtheta_ref)

            datum = caffe.io.array_to_datum(one_label.astype(np.float))
            str_id = '{:010}'.format(self.m_val_line_counter + i*2+1)
            val_label_txn.put(str_id, datum.SerializeToString())


            if i>0 and i%COMMIT_LENGTH==0:
                # commit data to disk every 1000 lines
                val_data_txn.commit()
                val_label_txn.commit()
                val_data_txn = val_data_env.begin(write=True)
                val_label_txn = val_label_env.begin(write=True)

        val_data_txn.commit()
        val_label_txn.commit()
        val_data_env.close()
        val_label_env.close()



        # insert into the test lmdb, including data & label ==================================
        test_data_env = lmdb.open(path+'test_data', map_size=map_size_data)
        test_data_txn = test_data_env.begin(write=True)

        test_label_env = lmdb.open(path+'test_label', map_size=map_size_label)
        test_label_txn = test_label_env.begin(write=True)

        for i in range(test_length - interval):
            # positive
            one_data = np.zeros((1, 2, 1081), dtype=np.float32)
            one_data[0, 0, :] = self.m_scan[i+train_length+val_length, :]
            one_data[0, 1, :] = self.m_scan[i+train_length+val_length + interval, :]

            datum = caffe.io.array_to_datum(one_data.astype(np.float))
            str_id = '{:010}'.format(self.m_test_line_counter+i*2)
            test_data_txn.put(str_id, datum.SerializeToString())

            # invert
            one_data = np.zeros((1, 2, 1081), dtype=np.float32)
            one_data[0, 1, :] = self.m_scan[i + train_length + val_length, :]
            one_data[0, 0, :] = self.m_scan[i + train_length + val_length + interval, :]

            datum = caffe.io.array_to_datum(one_data.astype(np.float))
            str_id = '{:010}'.format(self.m_test_line_counter + i*2+1)
            test_data_txn.put(str_id, datum.SerializeToString())


            # calculate the relative dx, dy, dtheta ----begin
            idx_prev = i+train_length+val_length
            idx_curr = i+train_length+val_length + interval
            yaw_prev_ref = self.m_reference_absolute[idx_prev, 2]
            x_prev_curr_ref = self.m_reference_absolute[idx_curr, 0] - self.m_reference_absolute[idx_prev, 0]
            y_prev_curr_ref = self.m_reference_absolute[idx_curr, 1] - self.m_reference_absolute[idx_prev, 1]
            theta_prev_curr_ref = self.m_reference_absolute[idx_curr, 2] - self.m_reference_absolute[idx_prev, 2]

            dx_ref = x_prev_curr_ref * math.cos(yaw_prev_ref) + y_prev_curr_ref * math.sin(yaw_prev_ref)
            dy_ref = y_prev_curr_ref * math.cos(yaw_prev_ref) - x_prev_curr_ref * math.sin(yaw_prev_ref)
            dtheta_ref = theta_prev_curr_ref
            # calculate the relative dx, dy, dtheta ----end

            # positive
            one_label = np.zeros((1, 1, 3), dtype=np.float32)
            one_label[0, 0, 0] = dx_ref
            one_label[0, 0, 1] = dy_ref
            one_label[0, 0, 2] = INPI(dtheta_ref)

            datum = caffe.io.array_to_datum(one_label.astype(np.float))
            str_id = '{:010}'.format(self.m_test_line_counter+i*2)
            test_label_txn.put(str_id, datum.SerializeToString())

            # invert
            one_label = np.zeros((1, 1, 3), dtype=np.float32)
            one_label[0, 0, 0] = -dx_ref * math.cos(dtheta_ref) - dy_ref * math.sin(dtheta_ref)
            one_label[0, 0, 1] = dx_ref * math.sin(dtheta_ref) - dy_ref * math.cos(dtheta_ref)
            one_label[0, 0, 2] = INPI(-dtheta_ref)

            datum = caffe.io.array_to_datum(one_label.astype(np.float))
            str_id = '{:010}'.format(self.m_test_line_counter + i * 2+1)
            test_label_txn.put(str_id, datum.SerializeToString())



            if i>0 and i%COMMIT_LENGTH==0:
                # commit data to disk every 1000 lines
                test_data_txn.commit()
                test_label_txn.commit()
                test_data_txn = test_data_env.begin(write=True)
                test_label_txn = test_label_env.begin(write=True)

        test_data_txn.commit()
        test_label_txn.commit()
        test_data_env.close()
        test_label_env.close()


        # increase the line counter, so that other frequency data can be added
        self.m_train_line_counter += (train_length-interval) * 2
        self.m_val_line_counter += (val_length-interval) * 2
        self.m_test_line_counter += (test_length-interval) * 2

        # calculate label mean and std
        self.m_label_dx_mean = np.mean(np.array(self.m_label_array)[:,0])
        self.m_label_dx_std = np.std(np.array(self.m_label_array)[:,0])
        self.m_label_dy_mean = np.mean(np.array(self.m_label_array)[:, 1])
        self.m_label_dy_std = np.std(np.array(self.m_label_array)[:, 1])
        self.m_label_dtheta_mean = np.mean(np.array(self.m_label_array)[:, 2])
        self.m_label_dtheta_std = np.std(np.array(self.m_label_array)[:, 2])

    def normalize_label(self, path):
        self.normalize_label_unit(path+'train_label', int(2 * 1024 * 1024 * 1024))
        self.normalize_label_unit(path + 'val_label', int(1 * 1024 * 1024 * 1024))
        self.normalize_label_unit(path + 'test_label', int(1 * 1024 * 1024 * 1024))

    def normalize_label_unit(self, input_filename, mapSize):
        # in-place normalization
        # get normalization parameter for label
        # this will be done ine npz2lmdb
        input_env = lmdb.open(input_filename, map_size=mapSize)
        input_txn = input_env.begin(write=True, buffers=True)
        line_number = input_env.stat()['entries']

        datum = caffe.proto.caffe_pb2.Datum()
        for i in range(line_number):
            datum.ParseFromString(input_txn.get('{:010}'.format(i)))

            datum.float_data[0] -= self.m_label_dx_mean
            datum.float_data[0] /= self.m_label_dx_std
            datum.float_data[1] -= self.m_label_dy_mean
            datum.float_data[1] /= self.m_label_dy_std
            datum.float_data[2] -= self.m_label_dtheta_mean
            datum.float_data[2] /= self.m_label_dtheta_std

            input_txn.put('{:010}'.format(i), datum.SerializeToString())

            if i > 0 and i % COMMIT_LENGTH == 0:
                input_txn.commit()
                input_txn = input_env.begin(write=True, buffers=True)

        input_txn.commit()
        input_env.close()

if __name__=="__main__":
    with Npz2lmdb_Invert('npz/my_log_15h.npz', 10) as npz2lmdb:
        t0 = time.time()

        npz2lmdb.add_noise_to_scan()
        npz2lmdb.normalize_scan()

        npz2lmdb.insert_data_into_lmdb(0.5, 'lmdb/')
        npz2lmdb.normalize_label('lmdb/')

        print npz2lmdb.m_train_line_counter
        print npz2lmdb.m_val_line_counter
        print npz2lmdb.m_test_line_counter

        print npz2lmdb.m_label_dx_mean
        print npz2lmdb.m_label_dx_std
        print npz2lmdb.m_label_dy_mean
        print npz2lmdb.m_label_dy_std
        print npz2lmdb.m_label_dtheta_mean
        print npz2lmdb.m_label_dtheta_std

        print('Time consumed (s): {0}'.format(time.time()-t0))


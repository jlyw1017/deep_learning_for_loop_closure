import numpy as np
import time
import lmdb
import sys
import math

caffe_root = '/data/jiaxin/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe


COMMIT_LENGTH = 10000

'''一个函数，作用是把角度调整位+-pai之间'''
def INPI(angle):
    if angle > math.pi:
        return angle - 2 * math.pi
    elif angle < -1 * math.pi:
        return angle + 2 * math.pi
    else:
        return angle

class Npz2lmdb:
    """Normalization & noise addition of scan will be done here. """
    def __init__(self, npz_filename, original_frequency):'''初始化需要一个文件名和一个频率'''
        self.m_original_frequency = original_frequency'''定义成员变量初始频率'''

        self.m_npz_filename = npz_filename'''定义成员变量文件名'''
        npz_file = np.load(npz_filename)'''打开npz文件'''

        self.m_scan = npz_file['scan']'''打开npz文件下面的，scan'''
        self.m_reference_absolute = npz_file['reference_absolute']'''打开npz文件下面的，reference_absolute'''
        self.m_scan_noisy = np.copy(self.m_scan)'''深复制scan'''

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

        print("Npz2lmdb ----------------")

    def __enter__(self):
        return self

    def __del__(self):
        del self.m_scan
        del self.m_reference_absolute

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_noise_to_scan(self):'''下面是给数据加上噪声'''
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
        train_length = int(math.floor(self.m_scan.shape[0]*0.8)) '''所有数据分组,数据总数*80%，取整维训练集长度，验证和测试各10%'''
        val_length = int(math.floor(self.m_scan.shape[0]*0.1))
        test_length = int(math.floor(self.m_scan.shape[0]*0.1))
        # train_length = 24 * 10000
        # val_length = 3 * 10000
        # test_length = 3 * 10000

        print('Size of scan (MByte): {0}'.format(float(self.m_scan.nbytes)/1000000))

        interval = int(round(self.m_original_frequency / frequency)) '''求整，10/0.5,每20个取一个数据'''

        # determine map_size of lmdb
        if 10*self.m_scan.nbytes>10000000:
            map_size_data = 20*self.m_scan.nbytes
        else:
            map_size_data = 20000000
        if 100*self.m_reference_absolute.nbytes>10000000:
            map_size_label = 200*self.m_reference_absolute.nbytes
        else:
            map_size_label = 20000000


        # insert into the train lmdb, including data & label ==================================
        train_data_env = lmdb.open(path+'train_data', map_size=map_size_data)
        train_data_txn = train_data_env.begin(write=True, buffers=True)

        train_label_env = lmdb.open(path+'train_label', map_size=map_size_label)
        train_label_txn = train_label_env.begin(write=True, buffers=True)

        for i in range(train_length-interval):'''遍历数据集，interval是取数据的间隔，减去是因为最后的那些数据没有可以匹配的了'''
            one_data = np.zeros((1,2,1081), dtype=np.float32) '''生成一个1*2*1081的矩阵'''
            one_data[0,0,:] = self.m_scan_noisy[i,:]
            one_data[0,1,:] = self.m_scan_noisy[i+interval,:]

            datum = caffe.io.array_to_datum(one_data.astype(np.float))
            str_id = '{:010}'.format(self.m_train_line_counter+i)
            train_data_txn.put(str_id, datum.SerializeToString())

            # calculate the relative dx, dy, dtheta ----begin
            ''' 坐标相减'''


            idx_prev = i
            idx_curr = i+interval
            yaw_prev_ref = self.m_reference_absolute[idx_prev, 2]
            x_prev_curr_ref = self.m_reference_absolute[idx_curr, 0] - self.m_reference_absolute[idx_prev, 0]
            y_prev_curr_ref = self.m_reference_absolute[idx_curr, 1] - self.m_reference_absolute[idx_prev, 1]  ''''''
            theta_prev_curr_ref = self.m_reference_absolute[idx_curr, 2] - self.m_reference_absolute[idx_prev, 2]

            dx_ref = x_prev_curr_ref * math.cos(yaw_prev_ref) + y_prev_curr_ref * math.sin(yaw_prev_ref)
            dy_ref = y_prev_curr_ref * math.cos(yaw_prev_ref) - x_prev_curr_ref * math.sin(yaw_prev_ref)
            dtheta_ref = theta_prev_curr_ref
            # calculate the relative dx, dy, dtheta ----end

            one_label = np.zeros((1,1,3), dtype=np.float32)
            one_label[0,0,0] = dx_ref
            one_label[0,0,1] = dy_ref
            one_label[0,0,2] = INPI(dtheta_ref)

            self.m_label_array.append([dx_ref, dy_ref, INPI(dtheta_ref)])

            datum = caffe.io.array_to_datum(one_label.astype(np.float))
            str_id = '{:010}'.format(self.m_train_line_counter+i)
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
            one_data = np.zeros((1, 2, 1081), dtype=np.float32)
            one_data[0, 0, :] = self.m_scan[i+train_length, :]
            one_data[0, 1, :] = self.m_scan[i+train_length + interval, :]

            datum = caffe.io.array_to_datum(one_data.astype(np.float))
            str_id = '{:010}'.format(self.m_val_line_counter+i)
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

            one_label = np.zeros((1, 1, 3), dtype=np.float32)
            one_label[0, 0, 0] = dx_ref
            one_label[0, 0, 1] = dy_ref
            one_label[0, 0, 2] = INPI(dtheta_ref)
            # self.m_label_array.append([dx_ref, dy_ref, INPI(dtheta_ref)])

            datum = caffe.io.array_to_datum(one_label.astype(np.float))
            str_id = '{:010}'.format(self.m_val_line_counter+i)
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
            one_data = np.zeros((1, 2, 1081), dtype=np.float32)
            one_data[0, 0, :] = self.m_scan[i+train_length+val_length, :]
            one_data[0, 1, :] = self.m_scan[i+train_length+val_length + interval, :]

            datum = caffe.io.array_to_datum(one_data.astype(np.float))
            str_id = '{:010}'.format(self.m_test_line_counter+i)
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

            one_label = np.zeros((1, 1, 3), dtype=np.float32)
            one_label[0, 0, 0] = dx_ref
            one_label[0, 0, 1] = dy_ref
            one_label[0, 0, 2] = INPI(dtheta_ref)
            # self.m_label_array.append([dx_ref, dy_ref, INPI(dtheta_ref)])

            datum = caffe.io.array_to_datum(one_label.astype(np.float))
            str_id = '{:010}'.format(self.m_test_line_counter+i)
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
        self.m_train_line_counter += (train_length-interval)
        self.m_val_line_counter += (val_length-interval)
        self.m_test_line_counter += (test_length-interval)

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
    with Npz2lmdb('npz/my_log_15h.npz', 10) as npz2lmdb:'''就是替换一下，省得输入命令行了'''
        t0 = time.time()

        npz2lmdb.add_noise_to_scan()'''调用加噪声的函数'''
        npz2lmdb.normalize_scan()'''调用正则的函数'''

        npz2lmdb.insert_data_into_lmdb(0.5, 'lmdb/')'''调用那个巨长的函数'''
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


import numpy as np
import struct

def convert_keys_to_npy(input_file, output_file):
    with open(input_file, 'rb') as f:
        # 读取向量的大小
        size_data = f.read(8)
        size = struct.unpack('Q', size_data)[0]

        keys = []
        for _ in range(size):
            # 读取每个元组的数据
            minKey_data = f.read(8)
            maxKey_data = f.read(8)
            level_data = f.read(4)
            order_data = f.read(4)

            minKey = struct.unpack('Q', minKey_data)[0]
            maxKey = struct.unpack('Q', maxKey_data)[0]
            level = struct.unpack('i', level_data)[0]
            order = struct.unpack('i', order_data)[0]

            keys.append((minKey, maxKey, level, order))

    keys = np.array(keys, dtype=np.uint64)
    np.save(output_file, keys)

if __name__ == "__main__":
    convert_keys_to_npy("sst_keys.dat", "sst_keys.npy")
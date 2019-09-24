from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf
import sys

_CKPT_200_FN = './checkpoint/rn-yolov3_B200.ckpt'
_CKPT_500_FN = './checkpoint/rn-yolov3_B500.ckpt'

#get_checkpoint = tf.train.latest_checkpoint(_CKPT_200_FN) 
#this retrieves the latest checkpoin file form path, but it also can be set manually

#inspect_list = tf.train.list_variables(get_checkpoint) 
#print(inspect_list)

if __name__ == '__main__':
	#main()
	fn = sys.argv[1]
	#print(fn)
	print_tensors_in_checkpoint_file(file_name=fn, tensor_name='', all_tensors=False)
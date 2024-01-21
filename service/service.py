from flask import Flask, request, redirect, url_for, Blueprint
import os
from flask_uploads import UploadSet, DEFAULTS, ARCHIVES, configure_uploads, UploadNotAllowed
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from threading import Thread
import utils

class Service(object):

    def __init__(self):
        # constants
        self.bottleneckDir = 'bottleneck/training'
        self.imagenetDir = 'imagenet'
        self.imagesDir = 'images'
        #self.BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
        self.BOTTLENECK_TENSOR_NAME = 'InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0'
        self.JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
        self.INPUT_TENSOR_NAME = 'input:0'
        self.RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
        self.FINAL_TRAINING_OPS = 'final_training_ops/weights/final_weights:0'

        # vars
        self.bottlenecks = None
        self.image_files = []
        self.bottleneck_tensor = None
        self.jpeg_data_tensor = None
        self.resized_input_tensor = None
        self.input_tensor = None
        self.final_ops_tensor = None
        self.sess = None

        self.blueprint = Blueprint('myapp', __name__, static_folder='images', template_folder=None)
        self.blueprint.add_url_rule('/upload', 'upload', self.upload_file, methods=['POST'])
        self.upload_folder = 'tmp'
        self.uploads_extensions=('txt', 'jpg', 'jpe', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff')
        self.uploaded_files = UploadSet('files', self.uploads_extensions)
        self.app = Flask('my app')
        self.app.config['UPLOADED_FILES_DEST'] = self.upload_folder
        configure_uploads(self.app, self.uploaded_files)
        self.app.register_blueprint(self.blueprint)

        self.create_inception_graph()
        self.start_load_bottlenecks()
        self.bottlenecks_loaded = False


    def start_load_bottlenecks(self):
        thread = Thread(target=Service.load_bottlenecks, args=(self,))
        thread.start()
        #thread.join()


    def load_bottlenecks(self):
        all_image_data = []
        batch_size = 100
        for dir in os.listdir(self.imagesDir)[:100]:
            for file in os.listdir(self.imagesDir + '/' + dir):
                full_name = self.imagesDir + '/' + dir + '/' + file
                self.image_files.append(full_name)

                #image_data = gfile.FastGFile(full_name, 'rb').read()
                image_data = utils.read_image(full_name)
                all_image_data.append(image_data)

                if (len(all_image_data) == batch_size):

                    bottleneck_values = self.run_bottleneck_on_image(self.sess, all_image_data, self.input_tensor)
                    bottleneck_values = bottleneck_values / (np.linalg.norm(bottleneck_values,
                                                    axis=1, ord=2)[:, None])

                    #bottleneck_values = np.expand_dims(bottleneck_values, 0)
                    if None is self.bottlenecks:
                        self.bottlenecks = bottleneck_values
                    else:
                        self.bottlenecks = np.append(self.bottlenecks, bottleneck_values, axis=0)

                    print("read bottlenecks of %s images" % str(len(self.bottlenecks)))

                    all_image_data = []

        self.bottlenecks = np.transpose(self.bottlenecks)
        self.bottlenecks_loaded = True


    def create_inception_graph(self):
        """"Creates a graph from saved GraphDef file and returns a Graph object.
        Returns:
          Graph holding the trained Inception network, and various tensors we'll be
          manipulating.
        """
        #with tf.Session() as sess:
        self.sess = tf.Session()
        model_filename = os.path.join(
            'imagenet', 'inception_v3_08_16.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            self.bottleneck_tensor, self.input_tensor  = (
                    tf.import_graph_def(graph_def, name='', return_elements=[
                     self.BOTTLENECK_TENSOR_NAME, self.INPUT_TENSOR_NAME
                     #self.RESIZED_INPUT_TENSOR_NAME,
                     #self.FINAL_TRAINING_OPS
                    ]))


    def run_bottleneck_on_image(self, sess, image_data, image_data_tensor):
        """Runs inference on an image to extract the 'bottleneck' summary layer.
        Args:
          sess: Current active TensorFlow Session.
          image_data: String of raw JPEG data.
          image_data_tensor: Input data layer in the graph.
          bottleneck_tensor: Layer before the final softmax.
        Returns:
          Numpy array of bottleneck values.
        """
        bottleneck_values = sess.run(
            self.bottleneck_tensor,
            {image_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values


    def _get_files_from_request(self, upload_set: UploadSet, path_prefix: str, key):
        """
        Return a list of full path to the uploaded files,
        if a field with `key` exists in the request.
        Otherwise return `None`.
        """
        if key in request.files:
            try:
                files = request.files.getlist(key)
                ret = []
                for s in files:

                    stored_filename = s.filename

                    # Use Flask check to verify if the file can be saved on the drive as-is
                    basename = upload_set.get_basename(s.filename)
                    if not upload_set.file_allowed(s, basename):
                        extension = os.path.splitext(s.filename)[1]
                        #stored_filename = '{}{}'.format(uuid.uuid4(), extension)
                        stored_filename = basename

                    saved = upload_set.save(s, folder='', name=stored_filename)
                    ret.append(stored_filename)
                return ret
            except UploadNotAllowed:
                print('upload not allowed')
        return None


    def upload_file(self):

        if self.bottlenecks_loaded == False:
            return 'bottlenecks are not loaded yet'

        files = self._get_files_from_request(self.uploaded_files, self.upload_folder, 'files')
        image_path = self.upload_folder + '/' + files[0]
        #image_data = gfile.FastGFile(image_path, 'rb').read()
        image_data = [utils.read_image(image_path)]
        bottleneck_values = self.run_bottleneck_on_image(self.sess, image_data, self.input_tensor)
        bottleneck_values = bottleneck_values / np.linalg.norm(bottleneck_values)
        dotprod = np.dot(bottleneck_values, self.bottlenecks)
        a = np.argsort(dotprod[:], axis=0)[-10:].tolist()
        similar_images = [self.image_files[i] for i in a]
        #delete file
        os.remove(image_path)
        return '\n'.join(similar_images)

    def run(self):
        port = os.getenv('VCAP_APP_PORT', '5000')
        self.app.run(host='0.0.0.0', port=int(port))


if __name__ == '__main__':
    Service().run()
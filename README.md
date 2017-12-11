# TensorFlow Model Deployment

A tutorial exploring multiple approaches to deploy / serve a trained TensorFlow (or Keras) model or multiple models 
in a production environment for prediction / inferences.

The code samples provided here may originally developed based on TensorFlow 1.2, 1.3 or 1.4. However, unless 
explicitly specified, they should work for all versions >= 1.0.

Table of Contents
=================
1.  [Import the Model Graph from Meta File](#importGraph)
2.  [Create the Model Graph from Scratch](#createGraph)
3.  [Restore Multiple Models](#restoreMultiple)
4.  [Freeze a Model before Serving it](#freeezeModel)
5.  [Convert a Keras model to a TensorFlow model](#convertKeras)
6.  [Deploy Multiple Freezed Models](#multiFreezed)
7.  [Serve a Model via Web Services](#webServices)

During the training, TensorFlow generates the following 3 files for each checkpoint, although optionally, 
you can choose not to create the meta file. You can ignore the file named checkpoint as it is not used in 
the prediction process.

1. meta file: It holds the compressed Protobufs graph of the model and all the other metadata associated, such 
as collections and operations.
2. index file: It holds an immutable table (key-value table) linking a serialised tensor name to where to find 
its data in the data file. 
3. data file: It is TensorBundle collection, which saves the values of all variables, such as weights.

### Import the Model Graph from Meta File
<a name="importGraph"></a>
One common approach is to restore the model graph from the meta file, and then restore weights and other data 
from the data file (index file will be used as well). Here is a sample code snippet:

```python
import tensorflow as tf
    
with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph("/trained/model_ckpt.meta")
    saver.restore(sess, "/trained/model_ckpt")
    
    # Retrieve Ops from the collection
        
    # Run sess to predict
```

A small trick here is where to place the following of code (saver) when you define the model graph for training. 
By default, only variables defined above this line will be saved into the meta file. If you don't plan to retrain
the model, you can leave the code defining your train_ops, such as optimizer, loss, accuracy below this line so 
that your model file can be reasonably smaller.

```
saver = tf.train.Saver()
```

You normally need to leave some hooks in the trained model so that you can easily feed the data for prediction. 
For example, you need to save logits and image_placehoder into the collection and save them in the training, and 
later retrieve them for prediction.

A concrete example can be found in train() and predict() methods 
[here](https://github.com/bshao001/DmsMsgRcg/blob/Sliding_Window_Version/misc/imgconvnets.py).

This applies to the case when the graph used for inference and training are the same or very similar. In case
the inference graph is very different from the graph used for training, this approach is not preferred as it 
would require the graph built for the training to adapt both training and inference, making it unnecessarily 
large.

### Create the Model Graph from Scratch
<a name="createGraph"></a>
Another common approach is to create the model graph from scratch instead of restoring the graph from the meta
file. This is extremely useful when the graph for inference is considerably different from the graph for training.
The new TensorFlow NMT model (https://github.com/tensorflow/nmt) is one of the cases.

```
import tensorflow as tf
# Replace this with your valid ModelCreator
import ModelCreator 
    
with tf.Session() as sess:
    # Replace this line with your valid ModelCreator and its arguments
    model = ModelCreator(training=False)
    # Restore model weights
    model.saver.restore(sess, "/trained/model_ckpt")
```

A concrete example can be found in the constructor (\_\_init\_\_ method) 
[here](https://github.com/bshao001/ChatLearner/blob/master/chatbot/botpredictor.py).

### Restore Multiple Models
<a name="restoreMultiple"></a>
Sometimes, you may need to load multiple trained models into a single TF session to work together for a task. For 
example, in a face recognition application, you may need a model to detect faces from a given images, then use 
another model to recognize these faces. In a typical photo OCR application, you normally require three models to 
work as a pipeline: model one to detect the text areas (blocks) from a given image; model two to segment characters 
from the text strings detected by the first model; and model three to recognize those characters.

Loading multiple models into a single session can be tricky if you don't do it properly. Here are the steps to follow:

1. For each of the models, you need to have a unique model_scope, and define all the variables within that scope when
building the graph for training:

```
with tf.variable_scope(model_scope):
    # Define variables here
```
 
2. At the time of restoring models, do the following:

```
tf.train.import_meta_graph(os.path.join(result_dir, result_file + ".meta"))
all_vars = tf.global_variables()
model_vars = [var for var in all_vars if var.name.startswith(model_scope)]
saver = tf.train.Saver(model_vars)
saver.restore(sess, os.path.join(result_dir, result_file))
```

Here, a TF session object (sess) is often passed into the method, as you don't want to create its own session here. 
Also, don't be fooled by the frequently used way of this statement:

```
saver = tf.train.import_meta_graph("/trained/model_ckpt.meta")
```

When the right side is run inside a TF session, the model graph is imported. It returns a saver, but you don't have 
to use it. My experience was if this saver is used to restore the data (weights), it won't work for loading multiple
models: it will complain all kinds of conflicts. 

A whole working example can be found in my [DmsMsgRcg](https://github.com/bshao001/DmsMsgRcg/tree/Sliding_Window_Version) 
project:
- Training: https://github.com/bshao001/DmsMsgRcg/blob/Sliding_Window_Version/misc/imgconvnets.py
- Predictor Definition: https://github.com/bshao001/DmsMsgRcg/blob/Sliding_Window_Version/misc/cnnpredictor.py
- Final Application: https://github.com/bshao001/DmsMsgRcg/blob/Sliding_Window_Version/mesgclsf/msgclassifier.py
 
### Freeze a Model before Serving it
<a name="freeezeModel"></a>
Sometimes, a trained model (file) can be very big, and ranging from half to several GB is a common case. At inference 
time, you don't have to deal with the big file if you choose to freeze the model. This process can normally decrease 
the model file to 20% to 30% of its original size, making the inference considerably faster.

Here are the 3 steps to achieve this:

1. Restore / load the trained model:

```
saver = tf.train.import_meta_graph("/trained/model_ckpt.meta")
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "/trained/model_ckpt")
```

2. Choose the output for the freezed model:

```
output_node_names = []
output_node_names.append("prediction_node")  # Specify the real node name
output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    input_graph_def,
    output_node_names
)
```

Here, you may need to use the following code to check the output node name:

```
for op in graph.get_operations():
    print(op.name)
```

Keep in mind that when you request to output an operation, all the other operations that it depends will also be 
saved. Therefore, you only need to specify the final output operation in the inference graph for freezing purpose.

3. Serialize and write the output graph and trained weights to the file system:

```
output_file = "model_file.pb"
with tf.gfile.GFile(output_file, "wb") as f:
    f.write(output_graph_def.SerializeToString())
    
sess.close()
```

A concrete working example, including how to use the freezed model for prediction can be found 
[here](https://github.com/bshao001/DmsMsgRcg/blob/master/misc/freezemodel.py).

### Convert a Keras model to a TensorFlow model
<a name="convertKeras"></a>

### Deploy Multiple Freezed Models
<a name="multiFreezed"></a>

### Serve a Model via Web Services
<a name="webServices"></a>
Although this does not directly relate to the problem of how to serve a trained model in TensorFlow, it is a 
commonly encountered issue. 

We train a machine learning model using python and TensorFlow, however, we often need to make use of the model 
to provide services to other different environments, such as a web application or a mobile application, or using 
different programming languages, such as Java or C#.

Both REST API and SOAP API can meet your needs on this. REST API is relatively light-weighted, but SOAP API is 
not that complicated either. You can pick any of them based on your personal preferences.   

- REST API

- SOAP API

### TensorFlow Serving

# References:
1. http://cv-tricks.com/how-to/freeze-tensorflow-models/
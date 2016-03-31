import numpy
import tensorflow as tf

import log
import nn
import nn.mlmc



# functions

def predict(train_data,
            test_data,
            hyper_params,
            experiment_setting,
            log_dir="tensorflow_events"):
  data_info = _analyze_data(train_data, test_data)

  x = tf.placeholder(
    tf.int32,
    (experiment_setting["batch_size"], data_info["document_length"]),
    name="x",
  )
  y_true = tf.placeholder(
    tf.int64,
    (experiment_setting["batch_size"], data_info["num_of_labels"]),
    name="y_true",
  )
  dropout_ratio = tf.placeholder(tf.float32, (), name="dropout_ratio")

  output_layer = nn.char2doc(
    x,
    char_space_size=data_info["char_space_size"],
    char_embedding_size=hyper_params["embedding_size"],
    dropout_ratio=dropout_ratio,
    hidden_layer_size=hyper_params["hidden_layer_size"],
    output_layer_size=data_info["num_of_labels"]*data_info["num_of_classes"],
  )
  error, accuracy, predicted_labels = nn.mlmc.classify(
    output_layer,
    y_true,
    num_of_labels=data_info["num_of_labels"],
  )

  train = tf.tuple((accuracy, error),
                   control_inputs=[tf.train.AdamOptimizer().minimize(error)])
  test = tf.tuple((accuracy, error, predicted_labels))

  with tf.Session() as session:
    logger = tf.train.SummaryWriter(log_dir, session.graph_def)
    session.run(tf.initialize_all_variables())

    for epoch in range(experiment_setting["num_of_epochs"]):
      log.message("epoch:", epoch)

      # train

      train_accuracy, train_error = session.run(train, {
        x : train_data.documents,
        t : train_data.labels,
        dropout_ratio : hyper_params["dropout_ratio"],
      })

      logger.add_summary(tf.scalar_summary(
        ["train_accuracy", "train_error"],
        [ train_accuracy ,  train_error ]
      ))

      # test

      test_accuracy, test_error, predicted_labels = session.run(test, {
        x : test_data.documents,
        t : test_data.labels,
        dropout_ratio : 0,
      })

      logger.add_summary(tf.scalar_summary(
        ["test_accuracy", "test_error"],
        [ test_accuracy ,  test_error ]
      ))

    return predicted_labels


def _analyze_data(train_data, test_data):
  assert train_data.documents.ndim == test_data.documents.ndim == 2 \
         and train_data.documents.shape[1] == test_data.documents.shape[1]
  assert train_data.labels.ndim == test_data.labels.ndim == 2 \
         and train_data.labels.shape[1] == test_data.labels.shape[1]

  all_documents = numpy.concatenate((train_data.documents,
                                     test_data.documents))
  all_labels = numpy.concatenate((train_data.labels, test_data.labels))

  return {
    "char_space_size" : all_documents.max() + 1,
    "document_length" : test_data.documents.shape[1],
    "num_of_labels" : test_data.labels.shape[1],
    "num_of_classes" : all_labels.max() + 1,
  }

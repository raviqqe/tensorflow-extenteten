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

  x = tf.placeholder(tf.int32, (None, data_info["document_length"]), name="x")
  y_true = tf.placeholder(tf.int32,
                          (None, data_info["num_of_labels"]),
                          name="y_true")

  error, accuracy, predicted_labels = nn.mlmc.classify(
      nn.my_net(x, hyper_params), y_true, data_info["num_of_labels"])

  train = tf.tuple((accuracy, error),
                   control_inputs=[tf.train.AdamOptimizer().minimize(error)])
  test = tf.tuple((accuracy, error, predicted_labels))

  logger = tf.train.SummaryWriter(log_dir, tf.get_default_graph())

  with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    for epoch in range(experiment_setting["num_of_epochs"]):
      log.message("epoch:", epoch)

      # train

      train_accuracy, train_error = session.run(train, {
        x : train_data.documents,
        t : train_data.labels,
      })

      logger.add_summary(tf.scalar_summary(
        ["train_accuracy", "train_error"],
        [ train_accuracy ,  train_error ]
      ))

      # test

      test_accuracy, test_error, predicted_labels = session.run(test, {
        x : test_data.documents,
        t : test_data.labels,
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
  all_labels = numpy.concatenate((train_data.labels, test_data.labels))

  return {
    "document_length" : test_data.documents.shape[1],
    "num_of_labels" : test_data.labels.shape[1],
    "num_of_classes" : all_labels.max() + 1,
  }

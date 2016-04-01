import numpy
import tensorflow as tf

import data
import log
import nn
import nn.mlmc



# functions

def predict(train_data,
            test_data,
            hyper_params,
            experiment_setting,
            summary_dir):
  data_info = _analyze_data(train_data, test_data)

  x = tf.placeholder(tf.int32, (None, data_info["document_length"]), name="x")
  y_true = tf.placeholder(tf.int64,
                          (None, data_info["num_of_labels"]),
                          name="y_true")
  dropout_ratio = tf.placeholder(tf.float32, (), name="dropout_ratio")

  output_layer = nn.char2doc(
    x,
    char_space_size=data_info["char_space_size"],
    char_embedding_size=hyper_params["embedding_size"],
    dropout_ratio=dropout_ratio,
    hidden_layer_size=hyper_params["hidden_layer_size"],
    output_layer_size=data_info["num_of_labels"]*data_info["num_of_classes"],
  )
  loss, accuracy, predicted_labels = nn.mlmc.classify(output_layer, y_true)

  do_training = tf.train.AdamOptimizer().minimize(loss)
  train_summary = tf.scalar_summary(["train_accuracy", "train_loss"],
                                    tf.pack([accuracy, loss]))
  test_summary = tf.scalar_summary(["test_accuracy", "test_loss"],
                                   tf.pack([accuracy , loss]))

  with tf.Session() as session:
    summarizer = tf.train.SummaryWriter(summary_dir, session.graph_def)
    session.run(tf.initialize_all_variables())

    for epoch in range(experiment_setting["num_of_epochs"]):
      log.message("epoch:", epoch)

      # train

      for batch in data.batches(data.shuffle(train_data),
                                experiment_setting["batch_size"]):
        session.run(do_training, {
          x : batch.documents,
          y_true : batch.labels,
          dropout_ratio : hyper_params["dropout_ratio"],
        })

      sampled_train_data = data.sample(train_data, test_data.size)

      summarizer.add_summary(session.run(
        train_summary, {
        x : sampled_train_data.documents,
        y_true : sampled_train_data.labels,
        dropout_ratio : 0,
      }), epoch)

      # test

      new_test_summary, last_predicted_labels = session.run(
        tf.tuple((test_summary, predicted_labels)), {
        x : test_data.documents,
        y_true : test_data.labels,
        dropout_ratio : 0,
      })
      summarizer.add_summary(new_test_summary, epoch)

    return last_predicted_labels


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

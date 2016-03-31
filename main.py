#!/usr/bin/env python

import argparse
import collections
import json
import numpy

import data
import model



# functions

def get_args():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("document_filename")
  arg_parser.add_argument("label_filename")
  arg_parser.add_argument("-l", "--predicted-labels",
                          dest="predicted_label_filename",
                          default=None)
  arg_parser.add_argument("-e", "--experiment-setting",
                          dest="experiment_setting_filename",
                          required=True)
  arg_parser.add_argument("-p", "--hyper-params",
                          dest="hyper_param_filename",
                          required=True)
  arg_parser.add_argument("-d", "--develop",
                          action="store_true")
  arg_parser.add_argument("-t", "--tensorflow-event-dir",
                          dest="summary_dir",
                          default="tensorflow_events")
  return arg_parser.parse_args()


def load_json_file(filename):
  with open(filename) as file:
    return json.load(file)


def load_hyper_params(filename):
  return load_json_file(filename)


def load_experiment_setting(filename):
  return load_json_file(filename)


def load_documents(filename):
  return numpy.load(filename)


def load_labels(filename):
  return numpy.load(filename).astype(numpy.int64)


def split_data(documents, labels, experiment_setting):
  data_sizes = (experiment_setting["train_data_size"],
                experiment_setting["develop_data_size"],
                experiment_setting["test_data_size"])

  assert documents.shape[0] == labels.shape[0] >= sum(data_sizes)

  train_documents, develop_documents, test_documents, _ \
      = numpy.split(documents, data_sizes)
  train_labels, develop_labels, test_labels, _ \
      = numpy.split(labels, data_sizes)

  return (data.Data(train_documents, train_labels),
          data.Data(develop_documents, develop_labels),
          data.Data(test_documents, test_labels))


def save_labels(labels, filename):
  numpy.savetxt(filename, labels, delimiter=",", fmt="%d")



# main routine

def main():
  args = get_args()

  experiment_setting \
      = load_experiment_setting(args.experiment_setting_filename)

  train_data, develop_data, test_data \
      = split_data(load_documents(args.document_filename),
                   load_labels(args.label_filename),
                   experiment_setting)

  predicted_labels = model.predict(
      train_data,
      develop_data if args.develop else test_data,
      hyper_params=load_hyper_params(args.hyper_param_filename),
      experiment_setting=experiment_setting,
      summary_dir=args.summary_dir)

  save_labels(predicted_labels, args.predicted_label_filename)


if __name__ == "__main__":
  main()

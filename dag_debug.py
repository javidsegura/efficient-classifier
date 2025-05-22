from efficient_classifier.utils.miscellaneous.dag import DAG


# Pipelines
pipelines = {
     "Pipeline1": ["Model1", "Model2"],
     "Pipeline2": ["Model3"]
}

# Phases
phases = {
      "Pipeline1": {
            "DataPreprocessing": {
                  "ClassImbalance": {
                        "_comment": "no thing was found in here",
                        "_subprocedures": {}
                  },
                  "Outliers": {
                        "_comment": None,
                        "_subprocedures": {}
                  },
                  "MissingValues": {
                        "_comment": None,
                        "_subprocedures": {}
                  },
                  "BoundChecking": {
                        "_comment": None,
                        "_subprocedures": {}
                  },
                  "FeatureScaling": {
                        "_comment": None,
                        "_subprocedures": {}
                  }
            },
            "FeatureAnalysis": { #phase
                  "Feature Selection": { # procedure
                        "_comment": None,
                        "_subprocedures": {
                              "Automatic": {
                                    "_comment": None,
                                    "_methods": {
                                          "Boruta": {"_comment": None},
                                          "L1": {"_comment": None}
                                    }
                              },
                              "Manual": {
                                    "_comment": None,
                                    "_methods": {}
                              }
                        }
                  },
                  "Feature Engineering": {
                        "_comment": None,
                        "_subprocedures": {}
                  },
                  "Feature Transformation": {
                        "_comment": None,
                        "_subprocedures": {}
                  }
            },
            "Modelling": {
            "pre-tuning": {
                  "_comment": None,
                  "_subprocedures": {
                        "feature Relevance": {
                              "_comment": None,
                              "_methods": {}
                        },
                        "Model Analysis": {
                              "_comment": None,
                              "_methods": {
                                    "accuracy": {"_comment": "1%"},
                                    "timeToFit": {"_comment": "2s"}
                              }
                        }
                  }
                  },
                  "in-tuning": {
                  "_comment": None,
                  "_subprocedures": {
                        "feature Relevance": {
                              "_comment": None,
                              "_methods": {}
                        },
                        "Model Analysis": {
                              "_comment": None,
                              "_methods": {
                                    "accuracy": {"_comment": "gb - 1%, rf - 2%"},
                                    "timeToFit": {"_comment": "gb - 2s, rf - 3s"}
                              }
                        }
                  }
                  },
                  "post-tuning": {
                  "_comment": None,
                  "_subprocedures": {
                        "Model Analysis": {
                              "_comment": "train_set = val_set + train_set",
                              "_methods": {
                                    "accuracy": {"_comment": "1%"},
                                    "timeToFit": {"_comment": "2s"}
                              }
                        },
                        "Model calibration": {
                              "_comment": "True",
                              "_methods": {}
                        }
                  }
                  }
            }
      },
      "Pipeline2": {
            "DataPreprocessing": {
                  "ClassImbalance": {
                        "_comment": "no thing was found in here",
                        "_subprocedures": {}
                  },
                  "Outliers": {
                        "_comment": None,
                        "_subprocedures": {}
                  },
                  "MissingValues": {
                        "_comment": None,
                        "_subprocedures": {}
                  },
                  "BoundChecking": {
                        "_comment": None,
                        "_subprocedures": {}
                  },
                  "FeatureScaling": {
                        "_comment": None,
                        "_subprocedures": {}
                  }
            },
            "FeatureAnalysis": { #phase
                  "Feature Selection": { # procedure
                        "_comment": None,
                        "_subprocedures": {
                              "Automatic": {
                                    "_comment": None,
                                    "_methods": {
                                          "Boruta": {"_comment": None},
                                          "L1": {"_comment": None}
                                    }
                              },
                              "Manual": {
                                    "_comment": None,
                                    "_methods": {}
                              }
                        }
                  },
                  "Feature Engineering": {
                        "_comment": None,
                        "_subprocedures": {}
                  },
                  "Feature Transformation": {
                        "_comment": None,
                        "_subprocedures": {}
                  }
            },
            "Modelling": {
            "pre-tuning": {
                  "_comment": None,
                  "_subprocedures": {
                        "feature Relevance": {
                              "_comment": None,
                              "_methods": {}
                        },
                        "Model Analysis": {
                              "_comment": None,
                              "_methods": {
                                    "accuracy": {"_comment": "1%"},
                                    "timeToFit": {"_comment": "2s"}
                              }
                        }
                  }
                  },
                  "in-tuning": {
                  "_comment": None,
                  "_subprocedures": {
                        "feature Relevance": {
                              "_comment": None,
                              "_methods": {}
                        },
                        "Model Analysis": {
                              "_comment": None,
                              "_methods": {
                                    "accuracy": {"_comment": "gb - 1%, rf - 2%"},
                                    "timeToFit": {"_comment": "gb - 2s, rf - 3s"}
                              }
                        }
                  }
                  },
            }
      },
}


obj = DAG(pipelines, phases)
obj.render()
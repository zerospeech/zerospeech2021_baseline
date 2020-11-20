import json
import argparse

import torch
from cpc.feature_loader import FeatureModule, loadModel
from cpc.criterion.clustering import kMeanCluster

from fairseq import tasks, checkpoint_utils
from fairseq.models.roberta import RobertaModel, RobertaHubInterface

def readArgs(pathArgs):
    print(f"Loading args from {pathArgs}")
    with open(pathArgs, 'r') as file:
        args = argparse.Namespace(**json.load(file))
    return args

def writeArgs(pathArgs, args):
    print(f"Writing args to {pathArgs}")
    with open(pathArgs, 'w') as file:
        json.dump(vars(args), file, indent=2)

def loadCPCFeatureMaker(pathCheckpoint, gru_level=-1, get_encoded=False, keep_hidden=True):
    """
    Load CPC Feature Maker from CPC checkpoint file.
    """
    # Set LSTM level
    if gru_level is not None and gru_level > 0:
        updateConfig = argparse.Namespace(nLevelsGRU=gru_level)
    else:
        updateConfig = None

    # Load CPC model
    model, nHiddenGar, nHiddenEncoder = loadModel([pathCheckpoint], updateConfig=updateConfig)
    
    # Keep hidden units at LSTM layers on sequential batches
    model.gAR.keepHidden = keep_hidden

    # Build CPC Feature Maker from CPC model
    featureMaker = FeatureModule(model, get_encoded=get_encoded)

    return featureMaker

def loadClusterModule(pathCheckpoint):
    """
    Load CPC Clustering Module from Clustering checkpoint file.
    """
    state_dict = torch.load(pathCheckpoint, map_location=torch.device('cpu'))
    clusterModule = kMeanCluster(torch.zeros(1, state_dict["n_clusters"], state_dict["dim"]))
    clusterModule.load_state_dict(state_dict["state_dict"])
    return clusterModule

def loadRobertaCheckpoint(pathBERTCheckpoint, pathData, from_pretrained=False):
    """
    Load Roberta model from checkpoint.
    If load a pretrained model from fairseq, set from_pretrained=True.
    """
    if from_pretrained: # Require connection to download bpe, possible errors for trained checkpoint that contains cfg 
        roberta = RobertaModel.from_pretrained(dirname(pathBERTCheckpoint), basename(pathBERTCheckpoint), pathData)
    else:
        # Set up the args Namespace
        model_args = argparse.Namespace(
            task='masked_lm',
            seed=-1,
            output_dictionary_size=-1,
            data=pathData,
            path=pathBERTCheckpoint
            )

        # Setup task
        task = tasks.setup_task(model_args)

        # Load model
        models, _model_args = checkpoint_utils.load_model_ensemble([model_args.path], task=task)
        model = models[0]

        # Wrap-up to RobertaHubInterface (to be consistent with RobertaModel.from_pretrained)
        roberta = RobertaHubInterface(_model_args, task, model)
    
    return roberta

def loadLSTMLMCheckpoint(pathLSTMCheckpoint, pathData):
    """
    Load lstm_lm model from checkpoint.
    """
    # Set up the args Namespace
    model_args = argparse.Namespace(
        task='language_modeling',
        output_dictionary_size=-1,
        data=pathData,
        path=pathLSTMCheckpoint
        )

    # Setup task
    task = tasks.setup_task(model_args)
    
    # Load model
    models, _model_args = checkpoint_utils.load_model_ensemble([model_args.path], task=task)
    model = models[0]
    
    return model, task
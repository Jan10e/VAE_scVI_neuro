import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.module.base import BaseModuleClass, LossRecorder
from scvi.nn import Encoder, DecoderSCVI
from scvi import settings
from typing import Dict, Optional, Tuple

class CustomSCVI(BaseModuleClass):
    """
    Custom scVI VAE model with modifications.
    
    This class extends the base scVI model with additional features:
    1. Modified encoder architecture with deeper layers
    2. Additional regularization parameters
    3. Custom dropout implementation
    """
    
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 3,  # Modified: Increased from default 1 or 2
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        gene_likelihood: str = "zinb",
        latent_distribution: str = "normal",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,  # New: Added layer normalization option
        **model_kwargs,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution
        
        # Modified encoder with deeper architecture
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,  # Using the new parameter
        )
        
        # Library size encoder
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        
        # Decoder
        self.decoder = DecoderSCVI(
            n_input,
            n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
        )
        
        # Batch effect module
        if n_batch > 0:
            self.batch_encoder = nn.Embedding(n_batch, n_latent)
        else:
            self.batch_encoder = None
            
        # New: Additional regularization parameter
        self.regularization_strength = model_kwargs.get("regularization_strength", 1.0)
            
    def _get_inference_input(self, tensors):
        x = tensors[settings.REGISTRY_KEYS.X_KEY]
        batch_index = tensors.get(settings.REGISTRY_KEYS.BATCH_KEY, None)
        
        return dict(x=x, batch_index=batch_index)
        
    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors.get(settings.REGISTRY_KEYS.BATCH_KEY, None)
        
        return dict(z=z, library=library, batch_index=batch_index)
    
    def inference(
        self, x, batch_index=None, n_samples=1
    ) -> Dict[str, torch.Tensor]:
        """
        High level inference method.
        
        Args:
            x: minibatch of data
            batch_index: batch indices
            n_samples: number of samples to draw per cell
            
        Returns:
            inference outputs dictionary
        """
        # Encoder outputs
        qz, z = self.z_encoder(x)
        ql, library = self.l_encoder(x)
        
        # Return outputs
        outputs = dict(
            z=z,
            qz=qz,
            library=library,
            ql=ql,
        )
        
        return outputs
    
    def generative(
        self, z, library, batch_index=None
    ) -> Dict[str, torch.Tensor]:
        """
        Generative model.
        
        Args:
            z: latent variable samples
            library: library size samples
            batch_index: batch indices
            
        Returns:
            generative outputs dictionary
        """
        # Add batch effect
        if self.batch_encoder is not None and batch_index is not None:
            batch_effect = self.batch_encoder(batch_index)
            z = z + batch_effect
            
        # Decoder
        px = self.decoder(z, library)
        
        return dict(px=px)
    
    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight=1.0,
    ) -> LossRecorder:
        """
        Loss computation.
        
        Args:
            tensors: data tensors
            inference_outputs: inference step outputs
            generative_outputs: generative step outputs
            kl_weight: KL divergence weight
            
        Returns:
            LossRecorder with all loss components
        """
        x = tensors[settings.REGISTRY_KEYS.X_KEY]
        
        qz = inference_outputs["qz"]
        ql = inference_outputs["ql"]
        px = generative_outputs["px"]
        
        # Reconstruction loss
        reconst_loss = -px.log_prob(x).sum(-1)
        
        # KL divergence
        kl_div_z = kl_weight * qz.kl(target=None).sum(-1)
        kl_div_l = kl_weight * ql.kl(target=None).sum(-1)
        
        # Modified: Additional regularization term
        latent_reg = self.regularization_strength * torch.norm(inference_outputs["z"], dim=1)
        
        # Record losses
        loss = torch.mean(reconst_loss + kl_div_z + kl_div_l + latent_reg)
        
        kl_local = dict(
            kl_div_z=kl_div_z,
            kl_div_l=kl_div_l,
        )
        
        return LossRecorder(
            loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local
        )

# Define a wrapper model class using our custom module
from scvi.model.base import BaseModelClass, RNASeqMixin, VAEMixin

class CustomSCVIModel(RNASeqMixin, VAEMixin, BaseModelClass):
    """
    A custom scVI model using our custom module.
    """

    def __init__(
        self,
        adata,
        n_hidden=128,
        n_latent=10,
        n_layers=3,
        dropout_rate=0.1,
        dispersion="gene",
        gene_likelihood="zinb",
        latent_distribution="normal",
        use_batch_norm=True,
        use_layer_norm=False,
        regularization_strength=1.0,
        **model_kwargs
    ):
        super().__init__(adata)
        
        # Build the model
        self.module = CustomSCVI(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            regularization_strength=regularization_strength,
            **model_kwargs
        )
        
        self._model_summary_string = "Custom scVI model with deeper architecture"
        self.init_params_ = self._get_init_params(locals())
import os
import typing
from typing import Any, Dict, List, Optional, Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata

if typing.TYPE_CHECKING:
    import mitie


class MitieNLP(Component):
    name = "nlp_mitie"

    provides = ["mitie_feature_extractor", "mitie_file"]

    defaults = {
        # name of the language model to load - this contains
        # the MITIE feature extractor
        "model": os.path.join("data", "total_word_feature_extractor.dat"),
    }

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 extractor=None
                 ) -> None:
        """Construct a new language model from the MITIE framework."""

        super(MitieNLP, self).__init__(component_config)

        self.extractor = extractor

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["mitie"]

    @classmethod
    def create(cls, cfg: RasaNLUModelConfig) -> 'MitieNLP':
        import mitie

        component_conf = cfg.for_component(cls.name, cls.defaults)
        model_file = component_conf.get("model")
        if not model_file:
            raise Exception("The MITIE component 'nlp_mitie' needs "
                            "the configuration value for 'model'."
                            "Please take a look at the "
                            "documentation in the pipeline section "
                            "to get more info about this "
                            "parameter.")
        extractor = mitie.total_word_feature_extractor(model_file)
        cls.ensure_proper_language_model(extractor)

        return MitieNLP(component_conf, extractor)

    @classmethod
    def cache_key(cls, model_metadata: Metadata) -> Optional[Text]:

        component_meta = model_metadata.for_component(cls.name)

        mitie_file = component_meta.get("model", None)
        if mitie_file is not None:
            return cls.name + "-" + str(os.path.abspath(mitie_file))
        else:
            return None

    def provide_context(self) -> Dict[Text, Any]:

        return {"mitie_feature_extractor": self.extractor,
                "mitie_file": self.component_config.get("model")}

    @staticmethod
    def ensure_proper_language_model(
            extractor: Optional['mitie.total_word_feature_extractor']) -> None:

        if extractor is None:
            raise Exception("Failed to load MITIE feature extractor. "
                            "Loading the model returned 'None'.")

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional['MitieNLP'] = None,
             **kwargs: Any
             ) -> 'MitieNLP':
        import mitie

        if cached_component:
            return cached_component

        component_meta = model_metadata.for_component(cls.name)
        mitie_file = component_meta.get("model")
        return cls(component_meta,
                   mitie.total_word_feature_extractor(mitie_file))

    def persist(self, model_dir: Text) -> Dict[Text, Any]:

        return {
            "mitie_feature_extractor_fingerprint": self.extractor.fingerprint,
            "model": self.component_config.get("model")
        }

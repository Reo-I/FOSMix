from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from typing import List, Dict, Union, Any, Optional
import torch

class DatasetConstants(BaseModel):
    classes: List[int]
    label_to_anno: Dict[int, List[int]]
    class_obj: Dict[int, str]

    @field_validator('label_to_anno', 'class_obj')
    def validate_length(cls, value, info: FieldValidationInfo) -> Dict[int, Union[str, List[int]]]:
        """Check the length of 'label_to_ano' and 'class_obj'"""
        if "classes" in info.data and len(info.data["classes"]) + 1 != len(value):
            raise ValueError("The length must be equal to the length of 'classes' + 1")
        return value
    
    @field_validator('label_to_anno')
    def check_RGB_scope(cls, label_to_anno) ->Dict[int, List[int]]:
        """Check whether RGB values are within the range of 0 to 255"""
        for rgb_v in label_to_anno.values():
            if not all(0<=v<=256  for v in rgb_v):
                raise ValueError(f"The value of RGB is {rgb_v}, which is outside the valid range of 0 to 255.")
        return label_to_anno

oem_meta_data = {
    "classes": list(range(1, 9)),
    "label_to_anno" : {
        0: [0, 0, 0], 
        1: [128, 0, 0], 
        2: [0, 255, 36],
        3: [148, 148, 148], 
        4: [255, 255, 255], 
        5: [34, 97, 38],
        6: [0, 69, 255], 
        7: [75, 181, 73], 
        8: [222, 31, 7]
    },
    "class_obj" : {
        0: "None", 
        1: "bareland", 
        2: "grass", 
        3: "pavement",
        4: "road", 
        5: "tree", 
        6: "water", 
        7: "cropland", 
        8: "building"
    }
}

flair_meta_data = {
    "classes": list(range(1, 13)),
    "label_to_anno": {
        0: [0, 0, 0], 
        1: [219, 14, 154], 
        2: [147, 142, 123], 
        3: [248, 12, 0], 
        4: [169, 113, 1], 
        5: [21, 83, 174], 
        6: [25, 74, 38], 
        7: [70, 228, 131], 
        8: [243, 166, 13], 
        9: [102, 0, 130], 
        10: [85, 255, 0], 
        11: [255, 243, 13], 
        12: [228, 223, 124]
    },
    "class_obj" : {
        0:"None", 
        1: "building", 
        2:"pervious surface", 
        3: "impervious surface", 
        4:"bare soil",
        5: "water", 
        6: "coniferous",
        7:"deciduous", 
        8: "brushwood", 
        9: "vineyard", 
        10: "herbaceous vegetation",
        11:"agricultural land",
        12:"plowed land"
    }
}

OEM = DatasetConstants(**oem_meta_data)
FLAIR = DatasetConstants(**flair_meta_data)

class ModelConfig:
    arbitrary_types_allowed = True

class BatchData(BaseModel):
    class Config(ModelConfig):
        pass
    x: torch.Tensor
    y: torch.Tensor
    fn: List[str]
    ref: Optional[torch.Tensor] = Field(default=None)
    color_x: Optional[torch.Tensor]= Field(default=None)
    domain: Optional[List[str]]= Field(default=None)
    shape: tuple
    domain: Optional[List[str]] = Field(default=None)
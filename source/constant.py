from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from typing import List, Dict, Union

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

flair_meta_data = ...

OEM = DatasetConstants(**oem_meta_data)
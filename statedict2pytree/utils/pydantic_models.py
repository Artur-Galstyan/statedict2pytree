from pydantic import BaseModel


class Field(BaseModel):
    path: str
    shape: tuple[int, ...]
    skip: bool = False


class TorchField(Field):
    pass


class JaxField(Field):
    type: str

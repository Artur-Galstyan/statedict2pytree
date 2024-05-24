from pydantic import BaseModel


class Field(BaseModel):
    path: str
    shape: tuple[int, ...]


class TorchField(Field):
    pass


class JaxField(Field):
    type: str

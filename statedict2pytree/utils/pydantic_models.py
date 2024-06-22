from pydantic import BaseModel


class ChunkifiedPytreePath(BaseModel):
    path: str


class ChunkifiedStatedictPath(BaseModel):
    path: str


class Field(BaseModel):
    path: str
    shape: tuple[int, ...]
    skip: bool = False


class TorchField(Field):
    pass


class JaxField(Field):
    type: str

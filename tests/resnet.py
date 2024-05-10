import equinox as eqx
import jax
from beartype.typing import Optional, Type
from equinox.nn import State
from jaxtyping import Array, PRNGKeyArray


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    *,
    key: PRNGKeyArray,
) -> eqx.nn.Conv2d:
    return eqx.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        use_bias=False,
        key=key,
    )


def conv1x1(
    in_channels: int, out_channels: int, stride: int = 1, *, key: PRNGKeyArray
) -> eqx.nn.Conv2d:
    return eqx.nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, use_bias=False, key=key
    )


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.BatchNorm

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, *, key: PRNGKeyArray
    ) -> None:
        self.conv = conv1x1(in_channels, out_channels, stride, key=key)
        self.norm = eqx.nn.BatchNorm(out_channels, axis_name="batch")

    def __call__(self, x: Array, state: State) -> tuple[Array, State]:
        x = self.conv(x)
        x, state = self.norm(x, state)

        return x, state


class BasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    downsample: Optional[Downsample]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[Downsample] = None,
        groups: int = 1,
        base_width: int = 64,
        *,
        key: PRNGKeyArray,
    ):
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        key, conv1_key, conv2_key = jax.random.split(key, 3)
        self.conv1 = conv3x3(in_channels, out_channels, stride, key=conv1_key)
        self.bn1 = eqx.nn.BatchNorm(out_channels, axis_name="batch")
        self.conv2 = conv3x3(out_channels, out_channels, key=conv2_key)
        self.bn2 = eqx.nn.BatchNorm(out_channels, axis_name="batch")
        self.downsample = downsample

    def __call__(self, x: Array, state: State) -> tuple[Array, State]:
        identity = x

        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)

        if self.downsample is not None:
            identity, state = self.downsample(x, state)

        out += identity
        out = jax.nn.relu(out)

        return out, state


class Bottleneck(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm

    downsample: Optional[Downsample]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[Downsample] = None,
        groups: int = 1,
        base_width: int = 64,
        *,
        key: PRNGKeyArray,
    ) -> None:
        width = int(out_channels * (base_width / 64.0)) * groups
        conv1_key, conv2_key, conv3_key = jax.random.split(key, 3)
        expansion = 4
        self.conv1 = conv1x1(in_channels, width, key=conv1_key)
        self.bn1 = eqx.nn.BatchNorm(width, axis_name="batch")
        self.conv2 = conv3x3(width, width, stride, groups, key=conv2_key)
        self.bn2 = eqx.nn.BatchNorm(width, axis_name="batch")
        self.conv3 = conv1x1(width, out_channels * expansion, key=conv3_key)
        self.bn3 = eqx.nn.BatchNorm(out_channels * expansion, axis_name="batch")
        self.downsample = downsample

    def __call__(self, x: Array, state: State) -> tuple[Array, State]:
        identity = x
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)

        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)

        x = self.conv3(x)
        x, state = self.bn3(x, state)

        if self.downsample is not None:
            identity, state = self.downsample(identity, state)

        x += identity
        x = jax.nn.relu(x)

        return x, state


class ResnetLayer(eqx.Module):
    layers: list[BasicBlock | Bottleneck]

    def __init__(self, layers: list[BasicBlock | Bottleneck]) -> None:
        self.layers = layers

    def __call__(self, x: Array, state: State) -> tuple[Array, State]:
        for l in self.layers:
            x, state = l(x, state)
        return x, state


class ResNet(eqx.Module):
    in_channels: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    maxpool: eqx.nn.MaxPool2d

    layer1: ResnetLayer
    layer2: ResnetLayer
    layer3: ResnetLayer
    layer4: ResnetLayer

    avgpool: eqx.nn.AdaptiveAvgPool2d
    fc: eqx.nn.Linear

    def __init__(
        self,
        block: Type[BasicBlock | Bottleneck],
        layers: list[int],
        image_channels: int = 3,
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.in_channels = 64
        key, conv_key = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(
            image_channels,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=conv_key,
        )
        self.bn1 = eqx.nn.BatchNorm(self.in_channels, axis_name="batch")
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        key, *layer_keys = jax.random.split(key, len(layers) + 1)
        self.layer1 = self._make_layer(
            block,
            out_channels=64,
            num_residual_blocks=layers[0],
            stride=1,
            groups=groups,
            base_width=width_per_group,
            key=layer_keys[0],
        )
        self.layer2 = self._make_layer(
            block,
            out_channels=128,
            num_residual_blocks=layers[1],
            stride=2,
            groups=groups,
            base_width=width_per_group,
            key=layer_keys[1],
        )

        self.layer3 = self._make_layer(
            block,
            out_channels=256,
            num_residual_blocks=layers[2],
            stride=2,
            groups=groups,
            base_width=width_per_group,
            key=layer_keys[2],
        )

        self.layer4 = self._make_layer(
            block,
            out_channels=512,
            num_residual_blocks=layers[3],
            stride=2,
            groups=groups,
            base_width=width_per_group,
            key=layer_keys[3],
        )

        self.avgpool = eqx.nn.AdaptiveAvgPool2d((1, 1))
        key, fc_key = jax.random.split(key)
        self.fc = eqx.nn.Linear(512 * _get_expansion(block), num_classes, key=fc_key)

    def __call__(self, x: Array, state: State) -> tuple[Array, State]:
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        x = self.maxpool(x)

        x, state = self.layer1(x, state)
        x, state = self.layer2(x, state)
        x, state = self.layer3(x, state)
        x, state = self.layer4(x, state)
        x = self.avgpool(x)
        x = x.reshape(-1)
        x = self.fc(x)
        return x, state

    def _make_layer(
        self,
        block: Type[BasicBlock | Bottleneck],
        out_channels: int,
        num_residual_blocks: int,
        stride: int,
        groups: int,
        base_width: int,
        *,
        key: PRNGKeyArray,
    ):
        downsample = None
        expansion = _get_expansion(block)
        key, downsample_key = jax.random.split(key)
        if stride != 1 or self.in_channels != out_channels * expansion:
            downsample = Downsample(
                self.in_channels, out_channels * expansion, stride, key=downsample_key
            )
        layers = []
        key, *layer_keys = jax.random.split(key, num_residual_blocks + 1)

        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                groups=groups,
                base_width=base_width,
                key=layer_keys[0],
            )
        )
        self.in_channels = out_channels * expansion
        for i in range(num_residual_blocks - 1):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    groups=groups,
                    base_width=base_width,
                    key=layer_keys[i + 1],
                )
            )
        return ResnetLayer(layers)


def _get_expansion(block_type: Type[Bottleneck | BasicBlock]) -> int:
    if block_type == Bottleneck:
        return 4
    else:
        return 1


def resnet18(
    image_channels: int = 3,
    num_classes: int = 1000,
    *,
    key: PRNGKeyArray,
    make_with_state: bool = True,
    **kwargs,
):
    layers = [2, 2, 2, 2]
    if make_with_state:
        return eqx.nn.make_with_state(ResNet)(
            BasicBlock, layers, image_channels, num_classes, **kwargs, key=key
        )
    else:
        return ResNet(
            BasicBlock, layers, image_channels, num_classes, **kwargs, key=key
        )


def resnet34(
    image_channels: int = 3, num_classes: int = 1000, *, key: PRNGKeyArray, **kwargs
):
    layers = [3, 4, 6, 3]
    return eqx.nn.make_with_state(ResNet)(
        BasicBlock, layers, image_channels, num_classes, **kwargs, key=key
    )


def resnet50(
    image_channels: int = 3,
    num_classes: int = 1000,
    *,
    key: PRNGKeyArray,
    make_with_state: bool = True,
    **kwargs,
):
    layers = [3, 4, 6, 3]
    if make_with_state:
        return eqx.nn.make_with_state(ResNet)(
            Bottleneck, layers, image_channels, num_classes, **kwargs, key=key
        )
    else:
        return ResNet(
            Bottleneck, layers, image_channels, num_classes, **kwargs, key=key
        )


def resnet101(
    image_channels: int = 3, num_classes: int = 1000, *, key: PRNGKeyArray, **kwargs
):
    layers = [3, 4, 23, 3]
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck, layers, image_channels, num_classes, **kwargs, key=key
    )


def resnet152(
    image_channels: int = 3, num_classes: int = 1000, *, key: PRNGKeyArray, **kwargs
):
    layers = [3, 8, 36, 3]
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck, layers, image_channels, num_classes, **kwargs, key=key
    )

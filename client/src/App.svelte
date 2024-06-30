<script lang="ts">
    //@ts-ignore
    import Sortable from "sortablejs/modular/sortable.complete.esm.js";
    import { onMount } from "svelte";
    import Swal from "sweetalert2";

    let model: string = "model.eqx";
    let anthropicModel: "opus" | "sonnet" | "sonnet3.5" | "haiku" = "haiku";

    const Toast = Swal.mixin({
        toast: true,
        position: "top-end",
        showConfirmButton: false,
        timer: 5000,
        timerProgressBar: true,
        didOpen: (toast) => {
            toast.onmouseenter = Swal.stopTimer;
            toast.onmouseleave = Swal.resumeTimer;
        },
    });
    type Field = {
        path: string;
        shape: Number[];
        skip: boolean;
        type: string | null;
    };

    let jaxFields: Field[] = [];
    let torchFields: Field[] = [];
    onMount(async () => {
        let req = await fetch("/startup/getJaxFields");
        jaxFields = (await req.json()) as Field[];
        req = await fetch("/startup/getTorchFields");
        torchFields = await req.json();
        setTimeout(() => {
            initSortable();
        }, 100);

        setTimeout(() => {
            onEnd();
        }, 500);
    });

    function initSortable() {
        new Sortable(document.getElementById("torch-fields"), {
            animation: 150,
            multiDrag: true,
            ghostClass: "bg-blue-400",
            selectedClass: "bg-accent",
            multiDragKey: "shift",
            onEnd: onEnd,
        });
    }

    function fetchJaxAndTorchFields() {
        let allTorchElements =
            document.getElementById("torch-fields")?.children;
        if (!allTorchElements) {
            Toast.fire({
                icon: "error",
                title: "Couldn't find PyTorch elements",
            });
            return {
                error: "Failed to fetch PyTorch elements",
                jaxFields: [],
                torchFields: [],
            };
        }
        let allTorchFields: HTMLElement[] = [];
        for (let i = 0; i < allTorchElements.length; i++) {
            allTorchFields.push(allTorchElements[i].firstChild as HTMLElement);
        }

        let newTorchFields: Field[] = [];
        allTorchFields.forEach((el) => {
            newTorchFields.push({
                path: el.getAttribute("data-path"),
                shape: el
                    .getAttribute("data-shape")
                    ?.split(",")
                    .map((x) => parseInt(x)),
                type: el.getAttribute("data-type"),
                skip: el.getAttribute("data-skip") === "true",
            } as Field);
        });

        const allJaxFields = document.querySelectorAll('[data-jax="jax"]');
        let newJaxFields: Field[] = [];
        allJaxFields.forEach((el) => {
            newJaxFields.push({
                path: el.getAttribute("data-path"),
                shape: el
                    .getAttribute("data-shape")
                    ?.split(",")
                    .map((x) => parseInt(x)),
                type: el.getAttribute("data-type"),
                skip: el.getAttribute("data-skip") === "true",
            } as Field);
        });

        return { jaxFields: newJaxFields, torchFields: newTorchFields };
    }

    function onEnd() {
        setTimeout(() => {
            const updatedFields = fetchJaxAndTorchFields();
            if (updatedFields.error) {
                Toast.fire({
                    icon: "error",
                    title: updatedFields.error,
                });
                return;
            }

            for (let i = 0; i < updatedFields.jaxFields.length; i++) {
                let jaxField = updatedFields.jaxFields[i];
                let torchField = updatedFields.torchFields[i];
                if (torchField === undefined) continue;
                if (torchField.skip === true) {
                    document
                        .getElementById("jax-" + i)
                        ?.classList.remove("bg-error");
                    continue;
                }
                let jaxShape = jaxField.shape;
                let torchShape = torchField.shape;
                //@ts-ignore
                let jaxShapeProduct = jaxShape.reduce((a, b) => a * b, 1);
                //@ts-ignore
                let torchShapeProduct = torchShape.reduce((a, b) => a * b, 1);
                if (jaxShapeProduct !== torchShapeProduct) {
                    document
                        .getElementById("jax-" + i)
                        ?.classList.add("bg-error");
                } else {
                    document
                        .getElementById("jax-" + i)
                        ?.classList.remove("bg-error");
                }
            }

            if (
                updatedFields.torchFields.length >
                updatedFields.jaxFields.length
            ) {
                for (
                    let i = updatedFields.jaxFields.length;
                    i < updatedFields.torchFields.length;
                    i++
                ) {
                    document
                        .getElementById("torch-" + i)
                        ?.classList.remove("bg-error");
                }
            }
        }, 100);
    }
    function checkFields(jaxFields: Field[], torchFields: Field[]) {
        if (jaxFields.length > torchFields.length) {
            return {
                error: "JAX and PyTorch have lengths! Make sure to pad the PyTorch side.",
            };
        }

        for (let i = 0; i < jaxFields.length; i++) {
            let jaxField = jaxFields[i];
            let torchField = torchFields[i];
            if (torchField.skip === true) {
                continue;
            }

            //@ts-ignore
            let jaxShapeProduct = jaxField.shape.reduce((a, b) => a * b, 1);
            //@ts-ignore
            let torchShapeProduct = torchField.shape.reduce((a, b) => a * b, 1);

            if (jaxShapeProduct !== torchShapeProduct) {
                return {
                    error: `JAX ${jaxField.path} with shape ${jaxField.shape} doesn't match PyTorch ${torchField.path} with shape ${torchField.shape}`,
                };
            }
        }
        return { success: true };
    }
    function removeSkipLayer(index: number) {
        torchFields = torchFields.toSpliced(index, 1);
        setTimeout(() => {
            onEnd();
        }, 100);
    }
    function addSkipLayer(index: number) {
        let fields = fetchJaxAndTorchFields();
        if (fields.error) {
            Toast.fire({
                icon: "error",
                text: fields.error,
            });
            return;
        }
        const newField = {
            skip: true,
            shape: [0],
            path: "SKIP",
            type: "SKIP",
        } as Field;
        torchFields = fields.torchFields.toSpliced(index, 0, newField);
        setTimeout(() => {
            onEnd();
        }, 100);
    }

    async function convert() {
        let fields = fetchJaxAndTorchFields();
        if (fields.error) {
            Toast.fire({
                icon: "error",
                title: fields.error,
            });
            return;
        }

        let check = checkFields(fields.jaxFields, fields.torchFields);
        if (check.error) {
            Toast.fire({
                icon: "error",
                title: "Failed to convert",
                text: check.error,
            });
            return;
        }

        const response = await fetch("/convert", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                model: model,
                jaxFields: fields.jaxFields,
                torchFields: fields.torchFields,
            }),
        });

        const res = await response.json();
        console.log(res);
        if (res.error) {
            Toast.fire({
                icon: "error",
                title: res.error,
            });
        } else {
            Toast.fire({
                icon: "success",
                title: "Conversion successful",
            });
        }
    }

    function padToMatch() {
        let fields = fetchJaxAndTorchFields();
        if (fields.error) {
            Toast.fire({
                icon: "error",
                text: fields.error,
            });
            return;
        }

        if (fields.torchFields.length < fields.jaxFields.length) {
            let toAdd = fields.jaxFields.length - fields.torchFields.length;
            for (let i = 0; i < toAdd; i++) {
                setTimeout(() => {
                    console.log("adding skip at ", i);
                    addSkipLayer(fields.jaxFields.length + i);
                }, 100);
            }
        }
    }

    function removeAllSkipLayers() {
        let fields = fetchJaxAndTorchFields();
        if (fields.error) {
            Toast.fire({
                icon: "error",
                text: fields.error,
            });
            return;
        }
        let filteredFields = [];
        for (let i = 0; i < fields.torchFields.length; i++) {
            if (fields.torchFields[i].skip === false) {
                filteredFields.push(fields.torchFields[i]);
            }
        }
        torchFields = filteredFields;
    }

    async function matchByName() {
        let fields = fetchJaxAndTorchFields();
        if (fields.error) {
            Toast.fire({
                icon: "error",
                text: fields.error,
            });
            return;
        }
        if (fields.jaxFields.length !== fields.torchFields.length) {
            Toast.fire({
                icon: "error",
                text: "PyTree and State Dict have diffent lengths. Make sure to pad first!",
            });
            return;
        }
        Toast.fire({
            icon: "info",
            title: "Matching by name...",
            text: "This can take a while! Hold tight.",
        });

        let content = `
You will get two lists of strings. These strings are fields of a JAX and PyTorch model.
For example:
--JAX START--
.layers[0].weight
.layers[1].weight
.layers[2].weight
.layers[3].weight
.layers[4].weight
--JAX END--

--PYTORCH START--
layers.0.weight
layers.1.weight
layers.4.weight
layers.2.weight
layers.3.weight
--PYTORCH END--

As you can see, the order doesn't match. This means, you should look at the PyTorch fields and
rearrange them, such that they match the JAX model. In the above example, the expected return value
is:
--PYTORCH START--
layers.0.weight
layers.1.weight
layers.2.weight
layers.3.weight
layers.4.weight
--PYTORCH END--

Here's another example:
--JAX START--
.conv1.weight
.bn1.weight
.bn1.bias
.bn1.state_index.init[0]
.bn1.state_index.init[1]
--JAX END--

--PYTORCH START--
bn1.running_mean
bn1.running_var
conv1.weight
bn1.weight
bn1.bias
--PYTORCH END--

The expected return value in this case is:

--PYTORCH START--
conv1.weight
bn1.weight
bn1.bias
bn1.running_mean
bn1.running_var
--PYTORCH END--


Sometimes, there are so-called "skip-layers" in the PyTorch model. Those can be put anywhere, preferably to
the end, because your priority is to match those fields that can be matched first! Here's an example:

--JAX START--
.layers[0].weight
.layers[1].weight
.layers[2].weight
.layers[3].weight
.layers[4].weight
--JAX END--

--PYTORCH START--
layers.0.weight
SKIP
layers.3.weight
layers.2.weight
layers.1.weight
--PYTORCH START--

This should return

--PYTORCH START--
layers.0.weight
layers.1.weight
layers.2.weight
layers.3.weight
SKIP
--PYTORCH START--


It's not always 100% which belongs to which. Use your best judgement. Start your response with
--PYTORCH START-- and end it with --PYTORCH END--.


Here's your input:
--JAX START--
        `;

        for (let i = 0; i < fields.jaxFields.length; i++) {
            content += fields.jaxFields[i].path + "\n";
        }
        content += "--JAX END--\n";
        content += "\n";
        content += "--PYTORCH START--\n";

        for (let i = 0; i < fields.torchFields.length; i++) {
            content += fields.torchFields[i].path + "\n";
        }

        content += "--PYTORCH END--";
        console.log(content);

        let req = await fetch("/anthropic", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                content: content,
                model: anthropicModel,
            }),
        });
        let res = await req.json();
        if (res.error) {
            Toast.fire({
                icon: "error",
                text: res.error,
            });
            return;
        }
        console.log(res);
        let responseContent = res.content;
        let lines = responseContent.split("\n");
        console.log(lines);
        let rearrangedTorchFields = [];
        for (let i = 0; i < lines.length; i++) {
            let matchingTorchField = fields.torchFields.find(
                (field) => field.path === lines[i],
            );
            if (matchingTorchField) {
                rearrangedTorchFields.push(matchingTorchField);
            }
        }
        if (fields.torchFields.length !== rearrangedTorchFields.length) {
            Toast.fire({
                icon: "error",
                text: "Some fields are missing in the response. Try a different model instead.",
            });
            return;
        }
        console.log("rearrangedTorchFields", rearrangedTorchFields);
        setTimeout(() => {
            torchFields = rearrangedTorchFields;
            onEnd();
            Toast.fire({
                icon: "success",
                title: "Success",
            });
        }, 500);
    }
</script>

<svelte:head><title>Statedict2PyTree</title></svelte:head>

<h1 class="text-3xl my-12">Welcome to Torch2Jax</h1>
<div class="my-4 flex justify-evenly">
    <button on:click={padToMatch} class="btn btn-accent">Pad To Match</button>
    <button on:click={removeAllSkipLayers} class="btn btn-secondary"
        >Remove All Skip Layers</button
    >
    <div>
        <button on:click={matchByName} class="btn btn-warning"
            >Match By Name</button
        >
        <select bind:value={anthropicModel}>
            <option value="opus">opus</option>
            <option value="sonnet">sonnet</option>
            <option value="sonnet3.5">sonnet3.5</option>
            <option value="haiku">haiku</option>
        </select>
    </div>
</div>
<div class="grid grid-cols-2 gap-x-2">
    <div class="">
        <h2 class="text-2xl">JAX</h2>
        <div id="jax-fields" class="">
            {#each jaxFields as field, i}
                <div
                    class="border h-12 rounded-xl flex flex-col justify-center"
                >
                    <div
                        id={"jax-" + String(i)}
                        class="whitespace-nowrap overflow-x-scroll cursor-pointer mx-2"
                        data-jax="jax"
                        data-path={field.path}
                        data-shape={field.shape}
                        data-skip={field.skip}
                        data-type={field.type}
                    >
                        {field.path}
                        {field.shape}
                    </div>
                </div>
            {/each}
        </div>
    </div>

    <div class="">
        <h2 class="text-2xl">PyTorch</h2>
        <div id="torch-fields" class="">
            {#key torchFields}
                {#each torchFields as field, i}
                    <div class="flex space-x-2 border h-12 rounded-xl">
                        <div
                            id={"torch-" + String(i)}
                            data-torch="torch"
                            data-path={field.path}
                            data-shape={field.shape}
                            data-skip={field.skip}
                            data-type={field.type}
                            class="flex-1 mx-2 my-auto whitespace-nowrap overflow-x-scroll cursor-pointer"
                        >
                            {#if field.skip}
                                SKIP
                            {:else}
                                {field.path}
                                {field.shape}
                            {/if}
                        </div>
                        {#if field.skip}
                            <button
                                class="btn btn-ghost"
                                on:click={() => {
                                    removeSkipLayer(i);
                                }}>-</button
                            >
                        {/if}
                        <button
                            class="btn btn-ghost"
                            on:click={() => {
                                addSkipLayer(i);
                            }}>+</button
                        >
                    </div>
                {/each}
            {/key}
        </div>
    </div>
</div>
<div class="flex justify-center my-12 w-full">
    <div class="flex flex-col justify-center w-full">
        <input
            id="name"
            type="text"
            name="name"
            class="input input-primary w-full"
            placeholder="Name of the new file (model.eqx per default)"
            bind:value={model}
        />
        <button
            on:click={convert}
            class="btn btn-accent btn-wide btn-lg mx-auto my-2"
        >
            Convert!
        </button>
    </div>
</div>

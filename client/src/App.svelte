<script lang="ts">
    //@ts-ignore
    import Sortable from "sortablejs/modular/sortable.complete.esm.js";
    import { onMount } from "svelte";
    import Swal from "sweetalert2";

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
    let torchSortable: Sortable;
    onMount(async () => {
        let req = await fetch("/startup/getJaxFields");
        jaxFields = (await req.json()) as Field[];
        req = await fetch("/startup/getTorchFields");
        torchFields = await req.json();
        setTimeout(() => {
            initSortable();
        }, 100);
    });

    function initSortable() {
        torchSortable = new Sortable(document.getElementById("torch-fields"), {
            animation: 150,
            multiDrag: true,
            ghostClass: "bg-blue-400",
            selectedClass: "bg-accent",
            multiDragKey: "shift",
            onEnd: onEnd,
        });
    }

    function swap(a: any, b: any, array: any[]) {
        const temp = array[a];
        array[a] = array[b];
        array[b] = temp;
        return array;
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
        const updatedFields = fetchJaxAndTorchFields();
        console.log(updatedFields);
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
                document
                    .getElementById("torch-" + i)
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
                document.getElementById("jax-" + i)?.classList.add("bg-error");
                document
                    .getElementById("torch-" + i)
                    ?.classList.add("bg-error");
            } else {
                document
                    .getElementById("jax-" + i)
                    ?.classList.remove("bg-error");
                document
                    .getElementById("torch-" + i)
                    ?.classList.remove("bg-error");
            }
        }

        if (updatedFields.torchFields.length > updatedFields.jaxFields.length) {
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
    }
    function checkFields() {}
    function removeSkipLayer(index: number) {
        console.log("removing skips at ", index);
        torchFields = torchFields.toSpliced(index, 1);
        setTimeout(() => {
            onEnd();
        }, 100);
    }
    function addSkipLayer(index: number) {
        console.log("adding skips at ", index);
        const newField = {
            skip: true,
            shape: [],
            path: "",
            type: "",
        } as Field;
        torchFields = torchFields.toSpliced(index, 0, newField);
        setTimeout(() => {
            onEnd();
        }, 100);
    }

    function convert() {}
</script>

<svelte:head><title>Statedict2PyTree</title></svelte:head>

<h1 class="text-3xl my-12">Welcome to Torch2Jax</h1>

<div class="grid grid-cols-2 gap-x-2">
    <div class="">
        <h2 class="text-2xl">JAX</h2>
        <div id="jax-fields" class="bg-base-200">
            {#each jaxFields as field, i}
                <div
                    id={"jax-" + String(i)}
                    class="whitespace-nowrap overflow-x-scroll cursor-pointer"
                    data-jax="jax"
                    data-path={field.path}
                    data-shape={field.shape}
                    data-skip={field.skip}
                    data-type={field.type}
                >
                    {field.path}
                    {field.shape}
                </div>
            {/each}
        </div>
    </div>

    <div class="">
        <h2 class="text-2xl">PyTorch</h2>
        <div id="torch-fields" class="bg-base-200">
            {#each torchFields as field, i}
                <div class="flex space-x-2">
                    <div
                        id={"torch-" + String(i)}
                        data-torch="torch"
                        data-path={field.path}
                        data-shape={field.shape}
                        data-skip={field.skip}
                        data-type={field.type}
                        class="flex-1 whitespace-nowrap overflow-x-scroll cursor-pointer"
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
                            on:click={() => {
                                removeSkipLayer(i);
                            }}>-</button
                        >
                    {/if}
                    <button
                        on:click={() => {
                            addSkipLayer(i);
                        }}>+</button
                    >
                </div>
            {/each}
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
            value="model.eqx"
        />
        <button
            on:click={convert}
            class="btn btn-accent btn-wide btn-lg mx-auto my-2"
        >
            Convert!
        </button>
    </div>
</div>

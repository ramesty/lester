import { useDispatch } from "react-redux";
import { Textarea, Field, Label, Description    } from "@headlessui/react";
import { InformationCircleIcon } from "@heroicons/react/24/outline"

function Input({index, value, inputValue, description, isValid}){

    const dispatch = useDispatch();

    return(
        // <div key={index} className="flex flex-col">
        //     <label className="mb-1 font-medium text-md text-white-700">
        //         {value}
        //     </label>
        //     <Textarea
        //         type="text"
        //         value={inputValue}
        //         onChange={(e) => dispatch(handleInputChange(index, e.target.value))}
        //         className={`p-2 rounded bg-neutral-800  ${isValid ? "border-gray-500" : "border-red-500"}`}
        //     />
        // </div>

        // <div className="w-full max-w-md p-1 border border-neutral-800 hover:bg-neutral-900 rounded text-center gap-2">
        //     <Field>
        //         <Label className="text-sm/6 font-medium text-gray-400">{value}</Label>
        //         <Description className="text-sm/6 text-white/50"></Description>
        //         <Textarea
        //         className={'mt-3 block w-full resize-none rounded-lg border-none bg-white/5 px-3 py-1.5 text-sm/6 text-gray focus:not-data-focus:outline-none data-focus:outline-2 data-focus:-outline-offset-2 data-focus:outline-white/25'}
        //         rows={2}
        //         defaultValue={inputValue}
        //         />
        //     </Field>
        // </div>

        <div className="w-full max-w-md p-1 border border-neutral-800 hover:bg-neutral-900 rounded text-center gap-2">
            <Field>
                <div className="items-center grid grid-cols-10">
                <Label className="font-medium text-gray-400 col-span-9 pr-2">{value}</Label>

                <div className="relative group col-span-1">
                    <InformationCircleIcon className="text-gray-500 hover:text-gray-600 cursor-pointer" />
                    <div className="fixed translate-x-8 -translate-y-12 bg-neutral-900 text-white p-4 rounded shadow-lg z-50 
                                    hidden group-hover:block">
                        {description}
                    </div>
                </div>
                </div>


                <Textarea
                className="mt-3 block w-full resize-none rounded-lg border-none bg-white/5 px-3 py-1.5 text-sm/6 text-gray 
                            focus:not-data-focus:outline-none data-focus:outline-2 data-focus:-outline-offset-2 data-focus:outline-white/25"
                rows={3}
                defaultValue={inputValue}
                />
            </Field>
        </div>

        // <Field className="flex flex-col">
        //     <Label className="data-disabled:opacity-50">{value}</Label>
        //     <Description className="data-disabled:opacity-50">Add any extra information about your event here.</Description>
        //     <Textarea name="description" className="data-disabled:bg-gray-100" value={inputValue}></Textarea>
        // </Field>
    )
}

export default Input
import { useDispatch } from "react-redux"
// import { Listbox } from "@headlessui/react"

function Dropdown({title, componentValue, setVal, componentLabels}){

    const dispatch = useDispatch()
    
    return(
        <div className="flex flex-col gap-2 p-1 border rounded border-neutral-900">
        <label>{title}</label>
        <select value={componentValue} onChange={(e) => dispatch(setVal(e.target.value))} className="p-2 border rounded bg-neutral-800">
            {componentLabels.map((value, index) =>(
            <option value={value} key={index} className="bg-neutral-700">{value}</option>
            ))}
        </select>
        </div>

    // <Listbox value={title} onChange={(e) => dispatch(setVal(e.target.value))}>
    //   <Listbox.Button>{componentValue}</Listbox.Button>
    //   <Listbox.Options>
    //     {componentLabels.map((val, idx) => (
    //       <Listbox.Option
    //         key={idx}
    //         value={val}
    //       >
    //         {val}
    //       </Listbox.Option>
    //     ))}
    //   </Listbox.Options>
    // </Listbox>
  
    )
}

export default Dropdown
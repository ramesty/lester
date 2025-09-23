import { useSelector } from "react-redux"

function DisplayBox(){

    const testResponse = useSelector((state)=>state.editor.testResponse)
    const submitErrorMsg = useSelector((state)=>state.editor.submitErrorMsg)

    return(
        <div className="row-span-1 row-start-5 overflow-y-auto border border-neutral-600 bg-neutral-900">

            <p className="mx-4 text-left">{testResponse}
            </p>

        </div>
    )
}

export default DisplayBox
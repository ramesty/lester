import { useSelector} from "react-redux";
import { useEditor } from "../context/editorContext";
import Input from "./Input";
import Dropdown from "./Dropdown";
import { setColour } from "../store/editorSlice";
import { colourLabels } from "../utils/labels";
import { useSubmissionActions} from "../utils/userSubmissions";


function InputPanel(){

    const inputs = useSelector((state) => state.editor.inputs);
    const valid = useSelector((state) => state.editor.valid);
    const colour = useSelector((state) => state.editor.colour);
    const { handleHighlight, clearHighlights} = useEditor();
    const { submitCode } = useSubmissionActions()

    const descriptions = [
        "These are the input files needed for the data preperations, the names of the files you read from.",
        "This is the input schema for each of the input files. Note, they should be seperated via nested arrays.",
        "These are the expected output columns from the data preperation stage of your pipeline. This is used to validate results.",
        "These are the input schema for the featurisation stage of your pipeline. I.e, what is the expected artifact input into featurisation.",
    ]

    return(
        <div className="grid grid-cols-1 gap-1 p-1 border border-neutral-800 rounded">

            <Input index={0} value="Data Preperation Input Arguments" inputValue={inputs[0]} description={descriptions[0]} isValid={valid[0]}/>
            <Input index={1} value="Data Preperation Input Schema" inputValue={inputs[1]} description={descriptions[1]} isValid={valid[1]}/>
            <Input index={2} value="Data Preperation Output Columns" inputValue={inputs[2]} description={descriptions[2]} isValid={valid[2]}/>
            <Input index={3} value="Featurisation Input Schema" inputValue={inputs[3]} description={descriptions[3]} isValid={valid[3]}/>
            <Dropdown title="Choose Highlight Colour" componentValue={colour} setVal={setColour} componentLabels={colourLabels} ></Dropdown>
            <button onClick={handleHighlight} className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded">Highlight</button>
            <button onClick={clearHighlights} className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded">Clear Highlights</button>
            <button onClick={submitCode} className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded">Synthesize</button>

        </div>        
    )
}

export default InputPanel
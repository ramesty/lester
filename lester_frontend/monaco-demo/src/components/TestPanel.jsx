import { useSelector } from "react-redux";
import Dropdown from "./Dropdown";
import {  setTestSelection, setRegenerateSelection } from "../store/editorSlice";
import {  testLabels, regenerateLabels } from "../utils/labels";
import { useEditor } from "../context/editorContext";
import { useSubmissionActions } from "../utils/userSubmissions";


function TestPanel(){

    const regenerateSelection = useSelector((state) => state.editor.regenerateSelection);
    const testSelection = useSelector((state) => state.editor.testSelection);
    const {handleHighlight, clearHighlights} = useEditor();
    const {testCodeSubmit, regenerateCodeSubmit } = useSubmissionActions();

    return(
        <div className="grid grid-cols-1 gap-1 p-1 border border-neutral-800 rounded">

            <Dropdown title="Test Synthesized Stage" componentValue={testSelection} setVal={setTestSelection} componentLabels={testLabels} ></Dropdown>
            <Dropdown title="Regenerate Synthesized Stage" componentValue={regenerateSelection} setVal={setRegenerateSelection} componentLabels={regenerateLabels} ></Dropdown>
            <button onClick={testCodeSubmit} className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded">Test Code</button>
            <button onClick={regenerateCodeSubmit} className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded">Regenerate Code</button>

        </div>        
    )
}

export default TestPanel
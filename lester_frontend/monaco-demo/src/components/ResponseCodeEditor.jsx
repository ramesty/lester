import Editor from "@monaco-editor/react";
import { useResponseEditor } from "../context/responseEditorContext";
import { useSelector } from "react-redux";

function ResponseCodeEditor(){

    const { handleResponseEditorDidMount, handleResponseEditorCodeChange } = useResponseEditor();
    const responseCode = useSelector((state)=>state.editor.responseCode)

    return(
        // <div className="flex flex-col h-full p-4 vs-bg-dark border border-neutral-600">
        <div className="col-start-2 col-span-1 p-4 vs-bg-dark border border-neutral-600">
          <h2 className="p-2 text-center text-lg">Synthesized Code</h2>
          <div className="h-full max-h-140">
            <Editor
              defaultLanguage="python"
              value={responseCode}
              onMount={handleResponseEditorDidMount}
              theme="vs-dark"
              options={{ readOnly: true, minimap: { enabled: false } }}
              height="100%"
            />
          </div>
        </div>
    )
}

export default ResponseCodeEditor
import Editor from "@monaco-editor/react";

function Codebox({validCode, title, code, handleCodeChange, isReadOnly, editorRef, monacoRef}){

    const handleEditorDidMount = (editor, monaco) => {

        console.log("editor mount")

        editorRef.current = editor;
        monacoRef.current = monaco;
    };

    return(
        <div className={`flex flex-col h-full p-2 border rounded ${validCode ? "border-gray-300" : "border-red-500"}`}>
          <h2>{title}</h2>
          <div className="flex-grow">
            <Editor
              defaultLanguage="python"
              defaultValue={code}
              onChange= {handleCodeChange}
              onMount={handleEditorDidMount}
              theme="vs-dark"
              options={{ readOnly: isReadOnly, minimap: { enabled: false } }}
              height="100%"
            />
          </div>
        </div>
    )
}

export default Codebox
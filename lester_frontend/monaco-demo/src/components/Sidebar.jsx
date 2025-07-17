import {useState} from "react"

function Sidebar({inputs, setInputs, colour, setColour, testSelection, setTestSelection, regenerateSelection, setRegenerateSelection, handleHighlight, clearHighlights, submitCode, testCodeSubmit, regenerateCodeSubmit, submitError, submitErrorMsg}){
    

    const labels = [
        "Dataprep input arg names",
        "Dataprep input schemas",
        "Dataprep output columns",
        "Featurisation input schema",
    ];

    const testLabels = [
        "no_selection",
        "test_data_preperation",
        "test_featurisation",
        "test_model"
    ]

    const regenerateLabels = [
        "no_selection",
        "regenerate_data_preperation",
        "regenerate_featurisation",
        "regenerate_model"
    ]

    const [valid, setValid] = useState(Array(labels.length).fill(true));

    const handleInputChange = (index, value) => {

        console.log("Changing inputs...")
        const updatedInputs = [...inputs];
        updatedInputs[index] = value;
        setInputs(updatedInputs);

    };

    return(
        <div className="col-span-1 flex flex-col gap-4 overflow-y-auto p-2 border rounded border-gray-300">
            <h2 className="text-xl font-semibold">Lester</h2>
            

            <div className="grid grid-cols-1 gap-2">
            {inputs.map((value, index) => (
                <div key={index} className="flex flex-col">
                <label className="mb-1 font-medium text-sm text-gray-700">
                    {labels[index]}
                </label>
                <input
                    type="text"
                    placeholder={value}
                    onChange={(e) => handleInputChange(index, e.target.value)}
                    className={`p-2 border rounded ${valid[index] ? "border-gray-300" : "border-red-500"}`}
                />
                </div>
            ))}
            </div>

            <div className="flex flex-col gap-2">
            <label>Choose Highlight Colour:</label>
            <select value={colour} onChange={(e) => setColour(e.target.value)} className="p-2 border rounded">
                <option value="green">Data Preparation - Green</option>
                <option value="yellow">Featurisation - Yellow</option>
                <option value="red">Model Training - Red</option>
            </select>
            </div>

            <div className="flex flex-col gap-2">
            <label>Test Synthesized Stage:</label>
            <select value={testSelection} onChange={(e) => setTestSelection(e.target.value)} className="p-2 border rounded">
                {testLabels.map((value, index) => (
                <option value={value} key={index}>{value}</option>
                ))}
            </select>
            </div>

            <div className="flex flex-col gap-2">
            <label>Regenerate Synthesized Stage:</label>
            <select value={regenerateSelection} onChange={(e) => setRegenerateSelection(e.target.value)} className="p-2 border rounded">
                {regenerateLabels.map((value, index) => (
                <option value={value} key={index}>{value}</option>
                ))}
            </select>
            </div>

            <div className="flex flex-col gap-2">
            <button onClick={handleHighlight} className="p-2 bg-blue-500 text-white rounded">Highlight Selection</button>
            <button onClick={clearHighlights} className="p-2 bg-gray-500 text-white rounded">Clear Highlights</button>
            <button onClick={submitCode} className="p-2 bg-green-600 text-white rounded">Submit Code</button>
            <button onClick={testCodeSubmit} className="p-2 bg-green-600 text-white rounded">Test Code</button>
            <button onClick={regenerateCodeSubmit} className="p-2 bg-green-600 text-white rounded">Regenerate Code</button>
            </div>

            {submitError && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded shadow-md z-50">
                <strong className="font-semibold">Input Error:</strong> {submitErrorMsg}
            </div>
            )}
        </div>
    )
}

export default Sidebar
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";

function App() {

  return(
    <BrowserRouter>
      <Routes>
        <Route>
            <Route>
                <Route path="/" element={<Home />} />
                <Route path="/test" element={<Home />} />
            </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App;

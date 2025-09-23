import Navbar from "../components/NavBar";

function PageLayout({children}){
    return(
        
        // <div className="h-screen w-screen max-h-screen max-w-screen grid grid-flow-col grid-rows-15 pr-2 pb-2">
        <div className="grid grid-rows-15 grid-cols-5 w-screen h-screen">
            {children}
        </div>
    )
}

export default PageLayout
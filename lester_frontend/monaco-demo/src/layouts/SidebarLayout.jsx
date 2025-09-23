function SidebarLayout({children}){

    return(
        // <div className="col-span-1 flex flex-col gap-4 overflow-y-auto p-2 bg-neutral-900 border border-neutral-600">
        <div className="col-start-1 row-start-2 overflow-y-auto row-span-full bg-neutral-900 border border-neutral-600 rounded">

            {children}

        </div>
    )
}

export default SidebarLayout
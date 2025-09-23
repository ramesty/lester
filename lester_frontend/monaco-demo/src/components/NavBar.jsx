import { useNavigate } from "react-router-dom";

export default function Navbar() {
  const navigate = useNavigate();

  return (
    <nav className="col-span-full row-start-1 grid grid-cols-6 items-center bg-neutral-900 text-white">
      <h3 className="col-span-1 text-xl font-semibold px-8">Lester</h3>

      <button
        className="col-span-1 col-start-6 mx-16 hover:text-sky-700 text-center"
        onClick={() => navigate("/about")}
      >
        About
      </button>
    </nav>
  );
}

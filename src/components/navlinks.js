import React from "react"
import { Link } from "gatsby"

import style from "./navlinks.module.css"

const NavLinks = ({ menuLinks }) => {
  return (
    <nav className={style.navigation}>
      <ul>
        {menuLinks.map(props => (
          <li key={props.name}>
            <Link to={props.link}>{props.name}</Link>
          </li>
        ))}
      </ul>
    </nav>
  )
}

export default NavLinks

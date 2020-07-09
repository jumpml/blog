import React from "react"
import { Link } from "gatsby"

import NavLinks from "./navlinks"
import style from "./header.module.css"

const Header = ({ siteTitle, siteDesc, menuLinks }) => (
  <header id="site-header" className={style.masthead} role="banner">
    <div className={style.masthead_info}>
      <Link to="/">
        <img
          src="/jumpML.svg"
          width="366"
          height="374"
          alt={siteTitle}
          className={style.site_logo}
        />
        <div className={style.site_title}>{siteTitle}</div>
        <div className={style.site_description}>{siteDesc}</div>
      </Link>
    </div>
    <NavLinks menuLinks={menuLinks} />
  </header>
)

export default Header

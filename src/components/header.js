import React from "react"
import { Link } from "gatsby"
import style from "./header.module.css"

const Header = ({ siteTitle, siteDesc }) => (
  <header id="site-header" className={style.masthead} role="banner">
    <div className={style.masthead_info}>
      <Link to="/">
        <img
          src="/jumpML.svg"
          width="100"
          height="100"
          alt={siteTitle}
          className={style.site_logo}
        />
        <div className={style.site_title}>{siteTitle}</div>
        <div className={style.site_description}>{siteDesc}</div>
      </Link>
    </div>
  </header>
)

export default Header

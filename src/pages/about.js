import React from "react"
import Layout from "../components/layout"
import SEO from "../components/seo"
import style from "./about.module.css"

const AboutPage = () => {
  return (
    <Layout>
      <SEO
        title="About JumpML"
        description="Information about JumpML"
        image="/jumpml.svg"
        pathname="/about"
        // Boolean indicating whether this is an article:
        // article
      />
      <section className={style.wrapper}>
        <h1 className={style.heading}>About Us</h1>
        <p>
          We help people build intelligent things that people love to use. We
          offer the following services
          <ol>
            <li> Training and teaching via Articles/Prototypes/Courses </li>
            <li> DSP/ML and Software consulting services </li>
            <li>
              {" "}
              Low-cost licensing of our ML/DSP algorithms for your projects
              (COMING SOON!)
            </li>
          </ol>
        </p>

        <p>
          {" "}
          Contact us at <a href="mailto:jumpml.com@gmail.com">
            JumpML email
          </a>{" "}
        </p>
      </section>
    </Layout>
  )
}

export default AboutPage
